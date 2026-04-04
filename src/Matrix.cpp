#include "Matrix.h"
#include "ErrorHandler.h"
#include "GlobalRand.h"
#include "NNStateRecorder.h"

Matrix::Matrix()
{
	this->m_numRows = 0;
	this->n_numCols = 0;
	this->arr = std::vector<double>();
}

Matrix::Matrix(const size_t& m_numR, const size_t& n_numC)
{
	verify_shape(m_numR, n_numC);

	this->m_numRows = m_numR;
	this->n_numCols = n_numC;
	this->arr = std::vector<double>();
	//this->arr = std::vector<double>(/*size*/ Nethis->m_numRows * this->n_numCols, /*default val*/ 0.0);

	this->arr.resize(static_cast<size_t>((this->m_numRows * this->n_numCols)), 0.0);
	this->arr.shrink_to_fit();
}

Matrix::Matrix(FileObjHandle& fp)
{
	FileObj::read(fp, &this->m_numRows);
	FileObj::read(fp, &this->n_numCols);
	
	this->arr = std::vector<double>(/*size*/ this->m_numRows * this->n_numCols, /*default val*/ 0.0);
	for (auto& elem : this->arr)
		FileObj::read(fp, &elem);

	verify_shape(this->m_numRows, this->n_numCols);
}

Matrix& Matrix::operator=(Matrix&& m) noexcept
{
	this->m_numRows = std::exchange(m.m_numRows, 0);
	this->n_numCols = std::exchange(m.n_numCols, 0);
	this->arr = std::move(m.arr);
	return *this;
}

void Matrix::WriteToFile(FileObjHandle& fp) const
{
	FileObj::write(fp, &this->m_numRows);
	FileObj::write(fp, &this->n_numCols);
	for (const auto& elem : this->arr)
		FileObj::write(fp, &elem);
}

const size_t& Matrix::getNumRows(const MatCond& matcond) const
{
	if (matcond == MatCond::Transpose)
		return this->n_numCols;
	else // if (matcond == MatCond::Normal)
		return this->m_numRows;

}

size_t Matrix::getNumElements() const
{
	return (this->n_numCols * this->m_numRows);
}

const size_t& Matrix::getNumCols(const MatCond& matcond) const
{
	if (matcond == MatCond::Transpose)
		return this->m_numRows;
	else //if (matcond == MatCond::Normal)
		return this->n_numCols;
}

const double& Matrix::getElement(const size_t& r, const size_t& c, const MatCond& matcond) const
{
	if (matcond == MatCond::Transpose)
	{
		try
		{
			return this->arr[c * this->n_numCols + r];
		}
		catch (...) { ErrorHandler::FatalError(); }
	}
	else //if (matcond == MatCond::Normal)
	{
		try
		{
			return this->arr[r * this->n_numCols + c];
		}
		catch (...) { ErrorHandler::FatalError(); }
	}
}

const double& Matrix::getLinElement(const size_t& ind) const
{
	try
	{
		return this->arr[ind];
	}
	catch (...) { ErrorHandler::FatalError(); }
}

void Matrix::setElement(const size_t& r, const size_t& c, const double& val, const MatCond& matcond)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
#endif
	if (matcond == MatCond::Transpose)
	{
		try
		{
			this->arr[c * this->n_numCols + r] = val;
		}
		catch (...) { ErrorHandler::FatalError(); }
	}
	else //if (matcond == MatCond::Transpose)
	{
		try
		{
			this->arr[r * this->n_numCols + c] = val;
		}
		catch (...) { ErrorHandler::FatalError(); }
	}
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
}

void Matrix::setLinElement(const size_t& ind, const double& val)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
#endif
	try
	{
		this->arr[ind] = val;
	}
	catch (...) { ErrorHandler::FatalError(); }
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
}

void Matrix::addLinElement(const size_t& ind, const double& val)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
#endif
	try
	{
		this->arr[ind] += val;
	}
	catch (...) { ErrorHandler::FatalError(); }
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
}

void Matrix::setRandDMatrix(std::uniform_real_distribution<double>* distribution)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
#endif
	for (auto& ind : this->arr)
		ind = GlobalRand::randRealDist(distribution);
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
}

void Matrix::setRandMatrix(std::uniform_int_distribution<int>* distribution)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
#endif
	for (auto& ind : this->arr)
		ind = GlobalRand::randIntDist(distribution);
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
}

void Matrix::setElementsToZero()
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
#endif
	for (auto& elem : this->arr)
		elem = 0;
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
}

auto Matrix::begin() const -> decltype(arr.begin())
{
	return arr.begin();
}

auto Matrix::end() const -> decltype(arr.end())
{
	return arr.end();
}

string Matrix::toString(const char& rbracket, const char& lbracket, const char& elemSep, const char& nwln, const MatCond& matcond) const
{
	auto matStr = string{rbracket};
	//auto rowLen = this->getNumRows(matcond);
	auto colLen = this->getNumCols(matcond);

	auto rowPos = size_t{ 0 };
	auto colPos = size_t{ 0 };
	auto indCnt = size_t{ 0 };
	for (auto& ind : this->arr)
	{
		if (colPos == 0)
		{
			if (rowPos > 0)
				matStr += nwln;
			//matStr += rbracket;
		}
		if (matcond == MatCond::Normal)
			matStr += std::to_string(ind);
		else
			matStr += std::to_string(this->getElement(rowPos, colPos, matcond));

		if (colPos == colLen - 1)
			;//matStr += lbracket;
		else
			matStr += elemSep;

		colPos = (colPos + 1) % colLen;
		if (colPos == 0)
			rowPos++;
		indCnt++;
	}
	matStr += lbracket;
	return matStr;
}

Matrix operator+(const Matrix& mat, const double& dbl)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
#endif
	auto nmat = Matrix(mat);
	for (auto& elem : nmat.arr)
		elem += dbl;
		
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix operator+(const Matrix& mat, const Matrix& mat2)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
	MatrixAPI::recordOperandA(mat2);
#endif
	auto nmat = Matrix(mat);
	if (nmat.m_numRows == mat2.m_numRows && nmat.n_numCols == mat2.n_numCols)
	{
		auto linInd = size_t{ 0 };

		for (auto& elem : nmat.arr)
			elem += mat2.arr[linInd++];
	}
	else
		ErrorHandler::FatalError();

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix& Matrix::operator+=(const Matrix& rhs)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
	MatrixAPI::recordOperandA(rhs);
#endif
	if (this->m_numRows == rhs.m_numRows && this->n_numCols == rhs.n_numCols)
	{
		auto linInd = size_t{ 0 };

		for (auto& elem : this->arr)
			elem += rhs.arr[linInd++];
	}
	else
		ErrorHandler::FatalError();
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
	return *this;
}

Matrix operator-(const Matrix& mat, const double& dbl)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
#endif
	auto nmat = Matrix(mat);
	for (auto& elem : nmat.arr)
		elem -= dbl;
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix operator-(const Matrix& mat, const Matrix& mat2)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
	MatrixAPI::recordOperandA(mat2);
#endif
	auto nmat = Matrix(mat);

	if (nmat.m_numRows == mat2.m_numRows && nmat.n_numCols == mat2.n_numCols)
	{
		auto linInd = size_t{ 0 };

		for (auto& elem : nmat.arr)
			elem -= mat2.arr[linInd++];
	}
	else
		ErrorHandler::FatalError();

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix operator*(const Matrix& mat, const double& dbl)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
#endif
	auto nmat = Matrix(mat);

	for (auto& elem : nmat.arr)
		elem *= dbl;

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix operator*(const Matrix& mat, const Matrix& mat2)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
	MatrixAPI::recordOperandA(mat2);
#endif
	auto nmat = Matrix(mat);

	if (nmat.m_numRows == mat2.m_numRows && nmat.n_numCols == mat2.n_numCols)
	{
		auto linInd = size_t{ 0 };

		for (auto& elem : nmat.arr)
			elem *= mat2.arr[linInd++];
	}
	else
		ErrorHandler::FatalError();

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix operator/(const Matrix& mat, const double& dbl)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
#endif
	auto nmat = Matrix(mat);

	if (dbl != 0)
	{
		for (auto& elem : nmat.arr)
			elem /= dbl;
	}
	else
	{
		for (auto& elem : nmat.arr)
			elem = nan("");
	}
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix operator/(const Matrix& mat, const Matrix& mat2)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
	MatrixAPI::recordOperandA(mat2);
#endif
	auto nmat = Matrix(mat);
	
	if (nmat.m_numRows == mat2.m_numRows && nmat.n_numCols == mat2.n_numCols)
	{
		auto linInd = size_t{ 0 };

		for (auto& elem : nmat.arr)
		{
			try
			{
				elem /= mat2.arr[linInd++];
			}
			catch (...) { elem = nan(""); };
		}
	}
	else
		ErrorHandler::FatalError();

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(nmat);
#endif
	return nmat;
}

Matrix& Matrix::operator*=(const double& rhs)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(*this);
#endif
	if (rhs != 0.0)
	{
		for (auto& elem : this->arr)
			elem *= rhs;
	}
	else
		ErrorHandler::FatalError();
		
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(*this);
#endif
	return *this;
}

void Matrix::Mult(Matrix& C, const Matrix& A, const Matrix& B, const MatCond& matcond)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(C);
	MatrixAPI::recordOperandA(A);
	MatrixAPI::recordOperandB(B);
#endif
	if (matcond == MatCond::Normal && A.n_numCols == B.m_numRows)
	{
		if (!(C.m_numRows == A.m_numRows && C.n_numCols == B.n_numCols))
			C = Matrix(A.m_numRows, B.n_numCols);
		else
			C.setElementsToZero();

		size_t sharedWdith = A.n_numCols; // = B.m_numRows
		size_t C_r = 0, C_c = 0;
		for (auto& ind : C.arr)
		{
			for (size_t it = 0; it < sharedWdith; it++)
				ind += A.getElement(C_r, it) * B.getElement(it, C_c);

			C_c = (C_c + 1) % C.n_numCols;
			if (C_c == 0)
				C_r++;
		}
	}
	else if (matcond == MatCond::Transpose && A.m_numRows == B.m_numRows)
	{
		if (!(C.m_numRows == A.n_numCols && C.n_numCols == B.n_numCols))
			C = Matrix(A.n_numCols, B.n_numCols);

		size_t sharedWdith = A.m_numRows; // = B.m_numCols
		size_t C_r = 0, C_c = 0;
		for (auto& ind : C.arr)
		{
			for (size_t it = 0; it < sharedWdith; it++)
				ind += A.getElement(it, C_r) * B.getElement(it, C_c);

			C_c = (C_c + 1) % C.n_numCols;
			if (C_c == 0)
				C_r++;
		}
	}
	else
	{
		ErrorHandler::FatalError();
	}

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(C);
#endif
}

Matrix Matrix::Mult(const Matrix& A, const Matrix& B, const MatCond& matcond)
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(A);
	MatrixAPI::recordOperandA(B);
#endif
	if (matcond == MatCond::Normal && A.n_numCols == B.m_numRows)
	{
		Matrix C = Matrix(A.m_numRows, B.n_numCols);

		size_t sharedWdith = A.n_numCols; // = B.m_numRows
		size_t C_r = 0, C_c = 0;
		for (auto& ind : C.arr)
		{
			for (size_t it = 0; it < sharedWdith; it++)
				ind += A.getElement(C_r, it) * B.getElement(it, C_c);

			C_c = (C_c + 1) % C.n_numCols;
			if (C_c == 0)
				C_r++;
		}

#if (DEBUG_NN_STATE_RECORDER)
		MatrixAPI::recordAfterOp(C);
#endif
		return C;
	}
	else if (matcond == MatCond::Transpose && A.m_numRows == B.m_numRows)
	{
		Matrix C = Matrix(A.n_numCols, B.n_numCols);

		size_t sharedWdith = A.m_numRows; // = B.m_n	umCols
		size_t C_r = 0, C_c = 0;
		for (auto& ind : C.arr)
		{
			for (size_t it = 0; it < sharedWdith; it++)
				ind += A.getElement(it, C_r) * B.getElement(it, C_c);

			C_c = (C_c + 1) % C.n_numCols;
			if (C_c == 0)
				C_r++;
		}

#if (DEBUG_NN_STATE_RECORDER)
		MatrixAPI::recordAfterOp(C);
#endif
		return C;
	}
	else
	{
		ErrorHandler::FatalError();
		return Matrix(0, 0);
	}
}

void Matrix::ApplyElementWiseFunc(Matrix& out, const Matrix& mat, double(*func)(const double&))
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(out);
	MatrixAPI::recordOperandA(mat);
#endif
	if (!(out.m_numRows == mat.m_numRows && out.n_numCols == mat.n_numCols))
		out = Matrix(mat.m_numRows, mat.n_numCols);

	auto linInd = size_t{ 0 };
	for (auto& elem : out.arr)
		elem = func(mat.arr[linInd++]);

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(out);
#endif
	return;
}

Matrix Matrix::ApplyElementWiseFunc(const Matrix& mat, double(*func)(const double&))
{
#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordBeforeOp(mat);
#endif
	Matrix out = Matrix(mat.m_numRows, mat.n_numCols);

	auto linInd = size_t{ 0 };
	for (auto& elem : out.arr)
		elem = func(mat.arr[linInd++]);

#if (DEBUG_NN_STATE_RECORDER)
	MatrixAPI::recordAfterOp(out);
#endif
	return out;
}

std::pair<size_t, size_t> Matrix::getMaxValuePos() const
{
	auto posOfMax = static_cast<size_t>(std::distance(this->arr.begin(), std::max_element(this->arr.begin(), this->arr.end())));
	auto ret = std::make_pair<size_t, size_t>(static_cast<size_t>(static_cast<double>(posOfMax) / this->n_numCols), posOfMax % this->n_numCols);

	return ret;
}

inline void Matrix::verify_shape(std::size_t r, std::size_t c)
{
	// guard overflow: r * c <= kMaxElements
	if (r > 0 && c > 0) {
		if (r > kMaxElements / c)
		{
			ErrorHandler::FatalError("Matrix: too many elements");
		}
	}
}