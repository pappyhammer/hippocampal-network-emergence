import os
from datetime import datetime
import yaml


def write_z_mvt_in_yaml(results_path, session_id, z_shifts):
    yaml_file = os.path.join(results_path, f"{session_id}_invalid_frames.yaml")
    frame_intervals_dict = dict()
    for z_shift in z_shifts:
        frame_intervals_dict[z_shift[0]] = z_shift[1]
    yaml_content = dict()
    yaml_content["frame_intervals"] = frame_intervals_dict
    yaml_content["as_invalid_times"] = True
    with open(yaml_file, 'w') as outfile:
        yaml.dump(yaml_content, outfile, default_flow_style=False)


def main():
    root_path = "/media/julien/Not_today/hne_not_today/"
    data_path = os.path.join(root_path, "data", "z_mvt_data")

    results_path = os.path.join(root_path, "results_hne/")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    file_name = "invalid_frames_z-mvt"
    session_id = None
    z_shifts = None
    if not os.path.isfile(os.path.join(data_path, file_name)):
        return
    with open(os.path.join(data_path, file_name), "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            if len(line) < 4:
                continue
            if line[0].lower() == "p":
                if session_id is not None and z_shifts is not None:
                    write_z_mvt_in_yaml(results_path, session_id, z_shifts)
                line = line.strip("\n")
                line = line.strip(" ")
                line = line.strip("*")
                line = line.strip(" ")
                session_id = line.lower()
                print(f"** Session_id: {session_id}")
                # if session_id not in ms_str_to_ms_dict:
                #     session_id = None
                #     z_shifts = None
                # else:
                z_shifts = []
            elif session_id is not None:
                split_values = line.split("-")
                if len(split_values) < 2:
                    print(f"line {line} {split_values}")
                if split_values[1].endswith("\n"):
                    split_values[1] = split_values[1][:-1]
                # removing one to be on Python indexing
                z_shifts.append((int(split_values[0])-1, int(split_values[1])-1))
        if session_id is not None and z_shifts is not None:
            write_z_mvt_in_yaml(results_path, session_id, z_shifts)


if __name__ == "__main__":
    main()
