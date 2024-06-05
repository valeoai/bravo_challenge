import argparse
import ast
import base64
import graphlib
import os
import sys


class ImportCollector(ast.NodeVisitor):
    def __init__(self, base_name):
        self.base_name = base_name
        self.modules = set()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith(f'{self.base_name}.'):
                self.modules.add(alias.name)

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module and node.module.startswith(f'{self.base_name}.'):
            self.modules.add(node.module)


def import_to_file_path(import_name, base_path):
    partial_path = os.path.sep.join(import_name.split('.'))
    if base_path:
        return os.path.join(base_path, partial_path + '.py')
    else:
        return partial_path + '.py'


def encode_module_contents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
        encoded = base64.b64encode(contents.encode('utf-8')).decode('utf-8')
        return encoded


def find_dependencies(base_path, base_name, module_import, import_graph=None):
    if import_graph is None:
        import_graph = {}
    elif module_import in import_graph:
        assert False, f'invalid recursive call: {module_import} in {import_graph}'  # this should never happen

    # Gets all immediate dependencies of the module in module_import
    collector = ImportCollector(base_name)
    module_path = import_to_file_path(module_import, base_path)
    with open(module_path, 'r', encoding='utf-8') as file:
        module_text = file.read()
    node = ast.parse(module_text, filename=module_path)
    collector.visit(node)

    # Adds the module to the import graph
    import_graph[module_import] = set(collector.modules)

    # Gets recursive dependencies
    for dep in collector.modules:
        if dep not in import_graph:
            find_dependencies(base_path, base_name, dep, import_graph)

    # Returns all dependencies
    return import_graph


def compile_to_single_script(base_path, base_name, entry_module):
    import_graph = find_dependencies(base_path, base_name, entry_module)
    import_order = graphlib.TopologicalSorter(import_graph).static_order()
    import_order = [mod for mod in import_order if mod != entry_module] + [entry_module]

    output_script = []
    output_script.append('import base64')
    output_script.append('import sys')
    output_script.append('\n')
    output_script.append('modules_data = {}')
    output_script.append('modules_path = {}')
    output_script.append('\n')

    for module in import_order:
        module_path = import_to_file_path(module, base_path)
        friendly_path = import_to_file_path(module, None)
        module_data = encode_module_contents(module_path)
        output_script.append(f"modules_path['{module}'] = '''{friendly_path}'''")
        output_script.append(f"modules_data['{module}'] = '''{module_data}'''\n")

    output_script.append(f"modules_order = [{', '.join([f'\"{mod}\"' for mod in import_order])}]\n")

    output_script.append('''
def load_module(module_name, main=False):
    if module_name not in modules_data:
        raise ImportError(f"No module named '{module_name}'")
    code = base64.b64decode(modules_data[module_name])
    module_namespace = {'__name__': '__main__' if main else module_name,
                        '__file__': modules_path[module_name]}
    exec(compile(code, module_namespace['__file__'], 'exec'), module_namespace)
    sys.modules[module_name] = type(sys)('module')
    sys.modules[module_name].__dict__.update(module_namespace)

if __name__ == '__main__':
    for mod in modules_order[:-1]:
        load_module(mod)
    load_module(modules_order[-1], main=True)
else:
    for mod in modules_order:
        load_module(mod)
''')

def main():
    parser = argparse.ArgumentParser(
         description='Exports a script with all internal dependencies resolved.',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base-path', help='path to the base directory of the submission', default='.')
    args = parser.parse_args()

    with open(args.results, 'wt', encoding='utf-8') as json_file:


if __name__ == '__main__':
    main()
