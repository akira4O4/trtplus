//
// Created by seeking on 6/5/24.
//
#include <utility>

#include "../3rdparty/yaml-cpp/include/yaml.h"
#include "iostream"
#ifndef MAIN_CONFIG_H
#define MAIN_CONFIG_H

class Config {
  public:
    Config() = default;
    explicit Config(std::string path);
    explicit Config(std::string path, bool auto_load = false);
    ~Config() = default;
    void                    load();
    void                    load(const std::string &path);
    template <typename T> T get(const std::string key) {
        return data_[ key ].as<T>();
    };
    void                       set_path(std::string path);
    template <typename T> void add(std::string key, T val) {
        data_[ key ] = val;
    };

  private:
    std::string path_;
    YAML::Node  data_;
};
#endif // MAIN_CONFIG_H
