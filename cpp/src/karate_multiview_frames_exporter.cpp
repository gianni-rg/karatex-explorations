#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

std::map<int, std::vector<nlohmann::json>> process_annotation(nlohmann::json anno) {
    std::map<int, std::vector<nlohmann::json>> frames;
    auto keypoints = anno["instance_info"];
    int num_frames = keypoints.size();

    for (int i = 0; i < num_frames; ++i) {
        int frame_id = keypoints[i]["frame_id"];
        frames[frame_id] = std::vector<nlohmann::json>();
        int num_persons_frame = keypoints[i]["instances"].size();
        std::vector<nlohmann::json> frame_data;
        int trackId = 0;

        for (int j = 0; j < num_persons_frame; ++j) {
            nlohmann::json person;
            person["bbox"] = keypoints[i]["instances"][j]["bbox"];
            person["trackid"] = trackId;
            person["keypoints"] = keypoints[i]["instances"][j]["keypoints"];
            frame_data.push_back(person);
        }

        frames[frame_id] = frame_data;
    }

    return frames;
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::string token;
    std::istringstream tokenStream(str);

    while (std::getline(tokenStream, token, delimiter)) {
        result.push_back(token);
    }

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_path> <clip_name>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string clip_name = argv[2];

    // Example:
    //std::string input_path = "D:\\Datasets\\karate\\Test";
    //std::string clip_name = "20230714_193412";

    std::string pose_folder = input_path + "\\" + clip_name;
    std::vector<std::string> pose_files;
    for (const auto & entry : fs::directory_iterator(pose_folder)) {
        if (entry.path().extension() == ".json") {
            pose_files.push_back(entry.path().string());
        }
    }

    std::map<std::string, std::map<int, std::vector<nlohmann::json>>> multiview_data;

    for (const auto & pose_file : pose_files) {
        std::ifstream f(pose_file);
        auto pose = nlohmann::json::parse(f);

        auto frames = process_annotation(pose);


        std::string camera_name = fs::path(pose_file).filename().string();
        auto camera_name_split = splitString(camera_name.substr(0, camera_name.find_last_of(".")), '_');
        camera_name = camera_name_split[1] + "_" + camera_name_split[2];

        multiview_data[camera_name] = frames;

        std::string output_folder_path = input_path + "\\" + clip_name + "\\" + camera_name;
        fs::create_directories(output_folder_path);

        for (const auto & frame : frames) {
            std::stringstream frameNumStr;
            frameNumStr << std::setw(6) << std::setfill('0') << frame.first;

            std::string output_file_name = output_folder_path + "\\" + frameNumStr.str() + ".json";
            std::ofstream outputFile(output_file_name);
            nlohmann::json frame_dic;
            frame_dic["frame_index"] = frame.first;
            frame_dic["person_data"] = frame.second;

            std::cout << "Exporting frame " << frame.first << " from camera " << camera_name << std::endl;

            outputFile << frame_dic << std::endl;
        }
    }

    return 0;
}