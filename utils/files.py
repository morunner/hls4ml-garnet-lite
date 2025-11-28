from pathlib import Path


def get_project_root_dir(project_name: str) -> Path:
    parts = list(Path.cwd().parts)
    for p in reversed(parts):
        if p == project_name:
            break
        else:
            parts.remove(p)
    return Path(*parts)


project_root = get_project_root_dir('hls4ml-garnet-lite')
data_path = project_root / 'data'
config_path = data_path / 'config'
model_path = data_path / 'models'
weights_path = data_path / 'weights'
dataset_path = data_path / 'dataset'
hls4ml_out_path = data_path / 'hls4ml_out'
