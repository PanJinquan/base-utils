//
// Created by dm on 2021/1/15.
//

#ifndef DETECTOR_FILE_UTILS_H
#define DETECTOR_FILE_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

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
 * 将string类型的内容保存为txt文件
 * @param path
 * @param data
 * @param bCover
 */
void write_datatxt(string path, string data, bool bCover = false);

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
 * @return
 */
string get_postfix(string path);

/***
 * 获得directory目录下的所有文件
 * @param dir
 * @return
 */
vector<string> get_files_list(string dir);


#endif //DETECTOR_FILE_UTILS_H
