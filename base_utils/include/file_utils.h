//
// Created by Pan on 2021/1/15.
//

#ifndef DETECTOR_FILE_UTILS_H
#define DETECTOR_FILE_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

#ifdef PLATFORM_ANDROID//_ANDROID
#define _ANDROID
#define separator "/"
#elif defined PLATFORM_LINUX//_LINUX
#define _LINUX
#define separator "/"
#elif defined PLATFORM_WINDOWS//_WIN32
#define _WINDOWS
#define separator  "\\"
#endif


/***
 * 保存vector<TYPE>的数据
 *     int num=128;
 *     string file="path/to/data.bin";
 *     vector<float> data(num, 0.0f);
 *     save_bin(file, data);
 * @tparam TYPE
 * @param path
 * @param data
 */
template<typename TYPE>
void save_bin(string path, vector<TYPE> &data) {
    ofstream file(path, ios::out | ios::binary);
    if (!file) {
        printf("Can't save file:%s\n", path.c_str());
        return;
    }
    file.write((char *) data.data(), sizeof(TYPE) * data.size());
    file.close();
    printf("finish save file:%s\n", path.c_str());

};

/***
 * 读取vector<TYPE>的数据
 *     int num=128;
 *     string file="path/to/data.bin";
 *     vector<float> out(num, 0.0f);
 *     load_bin(file, out);
 * @tparam TYPE
 * @param path
 * @param out
 */
template<typename TYPE>
void load_bin(string path, vector<TYPE> &out) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        printf("Can't open file:%s\n", path.c_str());
        return;
    }
    //vector<TYPE> data(size, 0.0f);
    file.read((char *) out.data(), sizeof(TYPE) * out.size());
    file.close();
    printf("finish load file:%s\n", path.c_str());
}

/***
 * 将string类型的内容保存为txt文件
 * @param path
 * @param data
 * @param bCover
 */
void write_datatxt(string path, string data, bool bCover = false);

/***
 * 保存vector<string>，每项保存一行
 * @param path
 * @param contents
 * @param bCover
 */
void write_contents(string path, vector<string> contents, bool bCover = true);

/***
 * 读取文本，每行一项
 * @param path
 * @return
 */
vector<string> read_contents(string path);


/***
 * 加载文件的内容
 * @param path
 * @return 以string形式，返回文件内容
 */
std::string load_file(string path);

/***
 * 加载文件的内容
 * @param path
 * @param file_string: 以string形式，返回文件内容
 * @return 0:表示读取成功，1表示读取失败
 */
int load_file(const char *path, std::string &file_string);


/***
 * 判断文件是否存在
 * @param path
 * @return
 */
bool file_exists(string path);


/***
 * 获得文件路径的文件名称
 * @param path
 * @return
 */
string get_basename(string path);

/***
 * 获得文件路径的父目录路径
 * @param path
 * @return
 */
string get_parent(string path);


/***
 * 获得文件路径的父目录文件名称
 * @param path
 * @return
 */
string get_subname(string path);


/***
 * 获得后缀名称
 * @param path
 * @param tolower 是否转为小写
 * @return
 */
string get_postfix(string path, bool tolower = true);


/***
 * 实现路径拼接
 * @param path1
 * @param path2
 * @return
 */
string path_joint(string path1, string path2);

/***
 * 删除文件
 * @param file
 * @return
 */
bool remove_file(string file);

/***
 * 获得directory目录下的所有文件
 * @param dir
 * @return
 */
vector<string> get_files_list(string dir);


#endif //DETECTOR_FILE_UTILS_H
