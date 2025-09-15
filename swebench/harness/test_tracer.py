import sys
import os
import ast
import json
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set

class DetailedCallGraphTracker:
    def __init__(self, output_file: str = "trace.jsonl"):
        self.call_stack = []
        self.call_graph = defaultdict(set)
        self.call_counts = defaultdict(int)
        self.max_depth = 0
        self.trace_data = []
        self.output_file = output_file
        self.source_cache = {}  # Cache for source file contents
        
        # Standard library modules to exclude
        self.stdlib_modules = {
            'abc', '_aix_support', '_android_support', 'annotationlib', 'antigravity', 
            '_apple_support', 'argparse', 'ast', '_ast_unparse', 'asyncio', 'base64', 
            'bdb', 'bisect', 'bz2', 'calendar', 'cmd', 'codecs', 'codeop', 'code', 
            'collections', '_collections_abc', '_colorize', 'colorsys', '_compat_pickle', 
            'compileall', 'compression', 'concurrent', 'configparser', 'contextlib', 
            'contextvars', 'copy', 'copyreg', 'cProfile', 'csv', 'ctypes', 'curses', 
            'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'doctest', 
            'email', 'encodings', 'ensurepip', 'enum', 'filecmp', 'fileinput', 'fnmatch', 
            'fractions', 'ftplib', 'functools', '__future__', 'genericpath', 'getopt', 
            'getpass', 'gettext', 'glob', 'graphlib', 'gzip', 'hashlib', 'heapq', 
            '__hello__', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'importlib', 
            'inspect', 'io', '_ios_support', 'ipaddress', 'json', 'keyword', 'linecache', 
            'locale', 'logging', 'lzma', 'mailbox', '_markupbase', 'mimetypes', 
            'modulefinder', 'multiprocessing', 'netrc', 'ntpath', 'nturl2path', 'numbers', 
            '_opcode_metadata', 'opcode', 'operator', 'optparse', 'os', '_osx_support', 
            'pathlib', 'pdb', '__phello__', 'pickle', 'pickletools', 'pkgutil', 
            'platform', 'plistlib', 'poplib', 'posixpath', 'pprint', 'profile', 
            'profiling', 'pstats', 'pty', '_py_abc', 'pyclbr', 'py_compile', 
            '_pydatetime', '_pydecimal', 'pydoc_data', 'pydoc', '_pyio', '_pylong', 
            '_pyrepl', '_py_warnings', 'queue', 'quopri', 'random', 're', 'reprlib', 
            'rlcompleter', 'runpy', 'sched', 'secrets', 'selectors', 'shelve', 'shlex', 
            'shutil', 'signal', '_sitebuiltins', 'site', 'smtplib', 'socket', 
            'socketserver', 'sqlite3', 'ssl', 'statistics', 'stat', 'string', 
            'stringprep', '_strptime', 'struct', 'subprocess', 'symtable', 'sysconfig', 
            'tabnanny', 'tarfile', 'tempfile', 'test', 'textwrap', 'this', 
            '_threading_local', 'threading', 'timeit', 'tkinter', 'tokenize', 'token', 
            'tomllib', 'traceback', 'tracemalloc', 'trace', 'tree', 'tty', 'turtledemo', 
            'turtle', 'types', 'typing', 'unittest', 'urllib', 'uuid', 'venv', 
            'warnings', 'wave', 'weakref', '_weakrefset', 'webbrowser', 'wsgiref', 
            'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zoneinfo'
        }
        
    def is_stdlib_call(self, filename):
        """Check if the call is from a standard library module"""
        if not filename:
            return False
            
        # Normalize the path
        normalized_path = os.path.normpath(filename)
        path_parts = normalized_path.split(os.sep)
        
        # Check if any part of the path matches stdlib modules
        for part in path_parts:
            if part in self.stdlib_modules:
                return True
                
        # Also check for common stdlib patterns
        if 'site-packages' in normalized_path:
            return False  # Third-party packages
            
        # Check if it's in the standard Python installation
        python_paths = [
            'lib/python',
            'Lib\\',
            '/usr/lib/python',
            '/usr/local/lib/python'
        ]
        
        for py_path in python_paths:
            if py_path in normalized_path:
                return True
                
        return False
        
    def get_source_line(self, filename: str, line_no: int) -> str:
        """Get the source code line from a file"""
        try:
            if filename not in self.source_cache:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.source_cache[filename] = f.readlines()
            
            if 1 <= line_no <= len(self.source_cache[filename]):
                return self.source_cache[filename][line_no - 1].rstrip()
            return ""
        except (IOError, OSError, UnicodeDecodeError):
            return ""
    
    def analyze_line_variables(self, source_line: str, frame) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Analyze which variables are read and written in a line using AST"""
        vars_read = {}
        vars_written = {}
        
        try:
            # Parse the line as an AST
            tree = ast.parse(source_line.strip(), mode='eval')
            
            # Extract variable names from the AST
            read_vars = set()
            written_vars = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Load):
                        read_vars.add(node.id)
                    elif isinstance(node.ctx, ast.Store):
                        written_vars.add(node.id)
                        
        except SyntaxError:
            # If it's not a valid expression, try as a statement
            try:
                tree = ast.parse(source_line.strip(), mode='exec')
                read_vars = set()
                written_vars = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        if isinstance(node.ctx, ast.Load):
                            read_vars.add(node.id)
                        elif isinstance(node.ctx, ast.Store):
                            written_vars.add(node.id)
            except SyntaxError:
                # If we still can't parse it, return empty sets
                read_vars = set()
                written_vars = set()
        
        # Get actual values from frame locals and globals
        frame_vars = {**frame.f_globals, **frame.f_locals}
        
        for var_name in read_vars:
            if var_name in frame_vars:
                vars_read[var_name] = self.serialize_value(frame_vars[var_name])
                
        for var_name in written_vars:
            if var_name in frame_vars:
                vars_written[var_name] = self.serialize_value(frame_vars[var_name])
                
        return vars_read, vars_written
    
    def serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON output"""
        try:
            # Try JSON serialization first
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            # Fall back to string representation
            return f"<non-serializable: {type(value).__name__}>"
    
    def get_function_parameters(self, frame) -> Dict[str, Any]:
        """Extract function parameters and their values"""
        code = frame.f_code
        param_names = code.co_varnames[:code.co_argcount]
        params = {}
        
        for name in param_names:
            if name in frame.f_locals:
                params[name] = self.serialize_value(frame.f_locals[name])
        
        # Handle *args and **kwargs
        if code.co_flags & 0x04:  # CO_VARARGS
            varargs_name = code.co_varnames[code.co_argcount]
            if varargs_name in frame.f_locals:
                params['*' + varargs_name] = self.serialize_value(frame.f_locals[varargs_name])
                
        if code.co_flags & 0x08:  # CO_VARKEYWORDS
            kwargs_index = code.co_argcount
            if code.co_flags & 0x04:  # also has *args
                kwargs_index += 1
            kwargs_name = code.co_varnames[kwargs_index]
            if kwargs_name in frame.f_locals:
                params['**' + kwargs_name] = self.serialize_value(frame.f_locals[kwargs_name])
                
        return params
    
    def get_current_function_name(self) -> str:
        """Get the name of the currently executing function"""
        if self.call_stack:
            return self.call_stack[-1]['func_name']
        return "<module>"
        
    def get_function_info(self, frame):
        """Extract detailed function information"""
        func_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        line_no = frame.f_lineno
        
        # Get just the filename without full path for cleaner output
        short_filename = os.path.basename(filename)
        
        # Get class name if this is a method
        class_name = None
        if 'self' in frame.f_locals:
            class_name = frame.f_locals['self'].__class__.__name__
        elif 'cls' in frame.f_locals:
            class_name = frame.f_locals['cls'].__name__
            
        if class_name:
            qualified_name = f"{short_filename}:{class_name}.{func_name}"
        else:
            qualified_name = f"{short_filename}:{func_name}"
            
        return {
            'qualified_name': qualified_name,
            'filename': filename,
            'short_filename': short_filename,
            'func_name': func_name,
            'class_name': class_name,
            'line_no': line_no
        }
    
    def add_trace_entry(self, event_type: str, frame, **kwargs):
        """Add a structured trace entry"""
        filename = frame.f_code.co_filename
        line_no = frame.f_lineno
        function_name = self.get_current_function_name()
        source_line = self.get_source_line(filename, line_no)
        
        entry = {
            'event_type': event_type,
            'line_number': line_no,
            'statement': source_line,
            'filepath': filename,
            'function_name': function_name,
            **kwargs
        }
        
        self.trace_data.append(entry)
        
    def trace_function(self, frame, event, arg):
        func_info = self.get_function_info(frame)
        
        # Skip standard library calls
        if self.is_stdlib_call(func_info['filename']):
            return self.trace_function
        
        if event == 'call':
            current_depth = len(self.call_stack)
            self.max_depth = max(self.max_depth, current_depth)
            
            # Get function parameters
            parameters = self.get_function_parameters(frame)
            
            # Record the relationship for call graph
            if self.call_stack:
                caller_info = self.call_stack[-1]
                caller_name = caller_info['qualified_name']
                callee_name = func_info['qualified_name']
                
                self.call_graph[caller_name].add(callee_name)
                self.call_counts[(caller_name, callee_name)] += 1
            
            self.call_stack.append(func_info)
            
            # Add function entry trace
            self.add_trace_entry(
                'Function', 
                frame, 
                parameters=parameters
            )
            
        elif event == 'return':
            if self.call_stack:
                returned_func = self.call_stack.pop()
                
                # Add return trace
                self.add_trace_entry(
                    'Return',
                    frame,
                    return_value=self.serialize_value(arg)
                )
                
        elif event == 'line':
            # Get source line and analyze variables
            source_line = self.get_source_line(frame.f_code.co_filename, frame.f_lineno)
            vars_read, vars_written = self.analyze_line_variables(source_line, frame)
            
            # Add line execution trace
            self.add_trace_entry(
                'Line',
                frame,
                vars_read=vars_read,
                vars_written=vars_written
            )
                
        elif event == 'exception':
            if self.call_stack:
                exc_type, exc_value, exc_tb = arg
                
                # Add exception trace
                self.add_trace_entry(
                    'Exception',
                    frame,
                    exception_type=exc_type.__name__,
                    exception_value=str(exc_value)
                )
                
        return self.trace_function
    
    def start_tracing(self):
        """Start the trace collection"""
        sys.settrace(self.trace_function)
        
    def stop_tracing(self):
        """Stop the trace collection"""
        sys.settrace(None)
        
    def save_trace(self):
        """Save the collected trace data to JSONL file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for entry in self.trace_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"Trace saved to {self.output_file}")
        
    def get_trace_summary(self):
        """Get a summary of the collected trace"""
        event_counts = defaultdict(int)
        for entry in self.trace_data:
            event_counts[entry['event_type']] += 1
            
        return {
            'total_events': len(self.trace_data),
            'event_breakdown': dict(event_counts),
            'max_call_depth': self.max_depth,
            'unique_functions': len(self.call_graph),
            'output_file': self.output_file
        }

# Example usage:
if __name__ == "__main__":
    # Example function to trace
    def example_function(x, y):
        a = x + y
        b = a * 2
        if b > 10:
            result = b - 5
        else:
            result = b + 5
        return result
    
    def another_function():
        values = [1, 2, 3, 4, 5]
        total = 0
        for val in values:
            total += val
        return total
    
    # Create and start the tracker
    tracker = DetailedCallGraphTracker("example_trace.jsonl")
    tracker.start_tracing()
    
    try:
        # Run some code to trace
        result1 = example_function(3, 7)
        result2 = another_function()
        print(f"Results: {result1}, {result2}")
        
    finally:
        # Always stop tracing and save results
        tracker.stop_tracing()
        tracker.save_trace()
        
        # Print summary
        summary = tracker.get_trace_summary()
        print("\nTrace Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")