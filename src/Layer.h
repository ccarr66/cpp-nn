#pragma once
#include "GLOBALS.h"
#include <memory>
#include "Matrix.h"
#include "FileObj.h"
#include "GlobalRand.h"

class NeuralNet;

class Layer : FileObj
{
public:

static constexpr const char* layerComponents[] = 
{
	"PrevLayerActivations",
	"Z_Activations",
	"Activations",
	"Weights",
	"Biases",
	"BiasCosts",
	"WeightCosts"
};
enum LayerComponentTypes
{
	eMinLayerComponentTypes = 0,
	ePrevLayerActivations = eMinLayerComponentTypes,
	eZ_Activations,
	eActivations,
	eWeights,
	eBiases,
	eBiasCosts,
	eWeightCosts,
	eMaxLayerComponentTypes
};

const Matrix& getLayerComponent(string component) const;
const Matrix& getLayerComponent(int component) const;

private:
NeuralNet* parentNN;
size_t layerIdx;

Matrix Z_Activations;
Matrix Activations;
Matrix Weights;
Matrix Biases;
Matrix BiasCosts;
Matrix WeightCosts;

public:
	Layer();
	Matrix* InitLayer(NeuralNet* nnet, const size_t layerIdx, FileObjHandle& fp);
	Matrix* InitLayer(NeuralNet* nnet, const size_t layerIdx, const size_t& layerSize);
	void WriteToFile(FileObjHandle&) const;

	void setRandDWB(std::uniform_real_distribution<double>*);
	void calculateActivations(double (*)(const double&), bool);

	const Matrix& getPrevLayerActivations(size_t layerIdx) const;
	const Matrix& getActivations() const;
	const Matrix& getZActivations() const;
	const Matrix& getWeights() const;
	const Matrix& getBiases() const;
	const Matrix& getError() const;
	const Matrix& getBiasCosts() const;
	const Matrix& getWeightCosts() const;
	
	size_t SetWeightsFromCostGradSection(const Matrix& costGrad, size_t offset);
	size_t SetBiasesFromCostGradSection(const Matrix& costGrad, size_t offset);

	void setError(Matrix&&);
};