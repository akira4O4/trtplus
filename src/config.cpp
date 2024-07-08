//
// Created by seeking on 6/5/24.
//
#include "config.h"

#include <utility>

Config::Config(std::string path) {
    path_ = std::move(path);
}
Config::Config(std::string path, bool auto_load) {
    path_ = std::move(path);
    if (auto_load) load();
}

void Config::load() {
    data_ = YAML::LoadFile(path_);
}
void Config::load(const std::string &path) {
    if (!path.empty()) path_ = path;
    data_ = YAML::LoadFile(path_);
}

void Config::set_path(std::string path) {
    path_ = std::move(path);
}