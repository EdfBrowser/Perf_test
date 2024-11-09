#ifndef DEFINES_H
#define DEFINES_H

#if defined(_WIN32) || defined(_WIN64)
#ifdef DLL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif
#else
#define DLL_API
#endif

#endif  // DEFINES_H