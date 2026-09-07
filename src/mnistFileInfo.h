#pragma once
#include "GLOBALS.h"
#include "string"

struct mnistFileInfo
{
	std::filesystem::path labelFile;
	std::filesystem::path imageFile;
};