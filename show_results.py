from typing import Optional, List
import json
from pathlib import Path
import re
import cmd
import argparse
import os
import numpy as np


from core import ObjectOrientedCausalGraph, EnvInfo
from scripts_ import get_env


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)

class ResultShower(cmd.Cmd):
    intro = "This program shows results saved in experiments."
    prompt = "show-result > "

    def __init__(self, root=None) -> None:
        super().__init__()
        
        self.__root: Optional[Path] = None
        if root is not None:
            self.__set_root(root)
        
        self.__filter: Optional[re.Pattern] = None
        self.__runs = self.__get_runs()
    
    def __set_root(self, root: str):
        _root = Path(root)
        if not _root.exists() or not _root.is_dir():
            print(f"{root} is not a existing directory.")
        else:
            self.__root = _root
    
    def __get_runs(self) -> List[str]:
        if self.__root is None:
            print("The root directory has not been set yet. Use 'root' command to set the root.")
            return []
        
        out = []
        for run_id in os.listdir(self.__root):
            if self.__filter is None or self.__filter.match(run_id):
                path = self.__root / run_id / 'results.json'
                if path.exists():
                    out.append(run_id)
        
        if len(out) == 0:
            print("No result files found in runs.")
        else:
            print("Found results.json in the following runs:")
            for run_id in out:
                print('-', run_id)
        
        return out
    
    def __readjson(self, path: Path, *keys: str):
        with path.open('r') as f:
            d = json.load(f)
        try:
            for k in keys:
                if not isinstance(d, dict):
                    raise TypeError
                else:
                    d = d[k]
            return d
        except KeyError:
            pass
        except TypeError:
            pass

        return None
    
    def __get_results(self, key: str):
        if key == '':
            keys = []
        else:
            keys = key.split('/')
        results = {}

        if self.__root is None:
            print("The root directory has not been set yet. Use 'root' command to set the root.")
            return results

        for run_id in self.__runs:
            path = self.__root / run_id / "results.json"
            results[run_id] = self.__readjson(path, *keys)

        return results
    
    def do_set_filter(self, arg: str):
        self.__filter = re.compile(arg)
        self.__runs = self.__get_runs()

    def do_unset_filter(self, arg: str):
        self.__filter = None
        self.__runs = self.__get_runs()
    
    def do_root(self, arg: str):
        self.__set_root(arg)
        self.__runs = self.__get_runs()
    
    def do_show(self, arg: str):
        '''
        show results of given keys.  
        usage: show key1/key2/.../keyn
        '''

        results = self.__get_results(arg)
        for run_id, value in results.items():
            print(run_id + ':', end='')
            value = str(value)
            if '\n' in value:
                print('\n\t', end='')
                print(value.replace('\n', '\n\t'))
            else:
                print(' ' + value)
    

    def __get_values_float(self, key: str):
        results = self.__get_results(key)
        values: List[float] = []
        for run_id, value in results.items():
            if value is None:
                continue
            try:
                values.append(float(value))
            except ValueError:
                value = str(value)
                if len(value) >= 20:
                    value = value[:20] + '...'
                print(f"Warning: fail to convert '{value}' to float.")
        return values

    def do_stat(self, arg: str):
        '''
        compute the statistic values of given keys.
        usage: stat key1/key2/.../keyn
        '''

        values = self.__get_values_float(arg)
        if len(values) == 0:
            print("No values.")
        else:
            print("mean:", np.mean(values))
            print("max:", np.max(values))
            print("min:", np.min(values))
            print("std:", np.std(values))
            print("median", np.median(values))
    
    def do_oocg(self, arg: str):
        if self.__root is None:
            print("The root directory has not been set yet. Use 'root' command to set the root.")
            return

        for run_id in self.__runs:
            path = self.__root / run_id / "args.json"
            env_id = self.__readjson(path, 'env', 'env_id')
            
            if not isinstance(env_id, str):
                print(f"Error in {run_id}: can not find env-id in args.json.")
                continue

            try:
                print(f"-----------------{run_id}-----------------")
                envcls = get_env(env_id)
                g = ObjectOrientedCausalGraph(EnvInfo(envcls.classes))
                state_dict = self.__readjson(self.__root / run_id / "causal-graph.json")
                assert isinstance(state_dict, dict)
                g.load_state_dict(state_dict)
                print(g)
                print("")
            except Exception as e:
                print(f"Error in {run_id}: {e}")

    def do_exit(self, arg: str):
        return True


parser.parse_args()
args = parser.parse_args()
ResultShower(args.root).cmdloop()
