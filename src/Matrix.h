#pragma once
#include "GLOBALS.h"
#include <vector>
#include <string>
#include <fstream>
#include "FileObj.h"
#include "GlobalRand.h"

enum class MatCond {Normal = 0, Transpose = 1};

class Matrix : FileObj
{
public:
static constexpr size_t MIN_LAYER_SIZE = 1;
static constexpr std::size_t kMaxElements = (1u << 24); // 68,719,476,736
static constexpr std::size_t max_elements() noexcept { return kMaxElements; }
private:
	std::vector<double> arr;
	size_t m_numRows, n_numCols;
	
	static inline void verify_shape(std::size_t r, std::size_t c);
public:

	Matrix();
	Matrix(const size_t& m, const size_t& n);
	Matrix(FileObjHandle& fp);
	Matrix(const Matrix&) = default;
	Matrix(Matrix&&) = default;
	Matrix& operator=(const Matrix&) = delete;
	Matrix& operator=(Matrix&&) noexcept;

	void WriteToFile(FileObjHandle&) const;

	size_t getNumElements() const;
	const size_t& getNumRows(const MatCond& = MatCond::Normal) const;
	const size_t& getNumCols(const MatCond& = MatCond::Normal) const;
	const double& getElement(const size_t& r, const size_t& c, const MatCond& = MatCond::Normal) const;
	const double& getLinElement(const size_t& ind) const;
	void setElement(const size_t& r, const size_t& c, const double& val, const MatCond& = MatCond::Normal);
	void setLinElement(const size_t& ind, const double& val);
	void addLinElement(const size_t& ind, const double& val);
	void setRandDMatrix(std::uniform_real_distribution<double>* );
	void setRandMatrix(std::uniform_int_distribution<int>* );
	void setElementsToZero();

	auto begin() const -> decltype(arr.begin());
	auto end() const -> decltype(arr.end());

	string toString(const char&, const char&, const char&, const char&, const MatCond& = MatCond::Normal) const;

	friend Matrix operator+(const Matrix&, const double&);
	friend Matrix operator+(const Matrix&, const Matrix&);
	Matrix& operator+=(const Matrix&);

	friend Matrix operator-(const Matrix&, const double&);
	friend Matrix operator-(const Matrix&, const Matrix&);

	friend Matrix operator*(const Matrix&, const double&);
	friend Matrix operator*(const Matrix&, const Matrix&);
	Matrix& operator*=(const double&);

	friend Matrix operator/(const Matrix&, const double&);
	friend Matrix operator/(const Matrix&, const Matrix&);

	static void Mult(Matrix&, const Matrix&, const Matrix&, const MatCond & = MatCond::Normal);
	static Matrix Mult(const Matrix&, const Matrix&, const MatCond & = MatCond::Normal);
	static void ApplyElementWiseFunc(Matrix&, const Matrix&, double (*)(const double&));
	static Matrix ApplyElementWiseFunc(const Matrix&, double (*)(const double&));

	std::pair<size_t/*row*/, size_t/*col*/> getMaxValuePos() const;

};