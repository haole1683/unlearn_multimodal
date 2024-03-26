from pathlib import Path

import json

def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True) 
    
def join_path(path, *paths):
    return Path(path).joinpath(*paths)


def record_result(result, folder_path):
    import os
    # file_path = os.path.join(folder_path, "result.txt")
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # with open(file_path, 'w') as f:
    #     for record in result:
    #         f.write(f'Epoch: {record["epoch"]}\n')
    #         f.write(f'Accuracy: {record["acc"]}\n')
    #         f.write(f'Class Accuracy: \n')
    #         for k,v in record['class_acc'].items():
    #             f.write(f'{k}: {v["correct_num"]},{v["total_num"]}, {v["correct_rate"]:.2f} | ')
    #         f.write('\n')
    
    # save as json
    file_path = os.path.join(folder_path, "result.json")
    with open(file_path, 'w') as f:
        json.dump(result, f)
        
def add_index_to_json_file(json_path):
    pass

def record_result():
    pass    