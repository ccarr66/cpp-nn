#pragma once
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>

#include "GLOBALS.h"
#include "NNStateCommon.h"
#include "Layer.h"
#include "Matrix.h"
#include "mnistFileInfo.h"
#if (DEBUG_NN_STATE_RECORDER)
#include "NNStateRecorder.h"
#endif

enum class CostFunction { Quadratic = 0 };
enum class ActivationFunction { Sigmoid = 0, ReLU = 1};

extern const std::vector<const char *> ActivationFunctionNames;
extern const std::vector<const char *> CostFunctionNames;

struct ProcessingRecord_t : FileObj
{
	size_t procIdx 						= 0;
	size_t procLabel 					= 0;
	size_t guess 		  				= 0;
	size_t percentCorrect 				= 0;
	size_t totalCount 					= 0;
	Matrix outputActivations			= {};
	std::vector<double> inputImage		= {};
	ProcessingRecord_t() 					= default;
	ProcessingRecord_t& operator=(const ProcessingRecord_t&);
	ProcessingRecord_t(FileObjHandle& fp);
	void WriteToFile(FileObjHandle&) const;
};
enum class ProcessingRecordDisplayCfg_T { inputImage = 0, outputActivations = 1, max = 2};
static constexpr const char* ProcessingRecordDisplayCfgStrs[ECBR_EcTy(ProcessingRecordDisplayCfg_T,max)] = {"inputImage", "outputActivations"};

struct NeuralNetParams_t {
    string filename;
    std::vector<size_t> layerSizes;
    size_t batchSize;
    int numBatches;
    double eta;
    mnistFileInfo trainingData;
    mnistFileInfo testData;
    ActivationFunction activationFunction;
	int activationFunctionSelect;
    CostFunction costFunction;
	int costFunctionSelect;

	//ctrl
	bool disableInitialWBRandomization = false;
};

struct NeuralNetDisplayState_t {
	uint32_t emptyForNow;
};

struct NeuralNetLabState_t;

class NeuralNet
{
public:

	static constexpr const char* NNComponents[] = 
	{
		"DesiredOutputs",
		"InputLayer",
		"OutputCostGrad",
		"WBCostGrad",
	};

	enum NNComponentTypes
	{
		eMinNNComponentTypes = 0,
		eDesiredOutputs = eMinNNComponentTypes,
		eInputLayer,
		eOutputCostGrad,
		eWBCostGrad,
		eMaxNNComponentTypes
	};

	const Matrix& getNNComponent(string component) const;
	const Matrix& getNNComponent(int component) const;

	enum class ProcessingTypes { None = 0, Training = 1, Testing = 2, Any = 3 };
private:
	string FileName;

	mnistFileInfo TrainingData;
	mnistFileInfo TestData;

	Matrix DesiredOutputs;

	Matrix InputLayer;
	std::vector<Layer> Layers;

	double Cost;
	Matrix OutputCostGrad;
	Matrix WBCostGrad;

	size_t BatchSize;
	int32_t NumBatches;
	size_t BatchesCompleted;

	size_t img_rows, img_cols;

	size_t inputTypeSize;

	double Eta;

	//dont need to save to disk
	mutable std::shared_mutex accessMutex; 
	bool tryToReadLayerNorm = false;
	std::vector<uint8_t> batchImgContainer;
	std::vector<uint8_t> batchLblContainer;
	std::unique_ptr<FileObjHandle> trainLblDataFP;
	std::unique_ptr<FileObjHandle> trainImgDataFP;
	std::unique_ptr<FileObjHandle> testLblDataFP;
	std::unique_ptr<FileObjHandle> testImgDataFP;
	std::vector<size_t> shuffledProcessingBatches;
	std::uniform_real_distribution<double> lclRandDistrbution;
	friend class NeuralNetProcessLifetimeManager;
	class NeuralNetProcessLifetimeManager
	{
	public:
		NeuralNet* parent;
		std::mutex processingMutex;               // Mutex to guard access to isProcessing
		ProcessingTypes currentProcess;
    	std::condition_variable processStoppedCV; // Condition variable for graceful shutdown
		bool isProcessing{ false };                 // Flag to check if processing is ongoing
		bool stopProcessing{ false };               // Flag to request stopping processing
	#if (!USE_THREADPOOL)
		std::thread processingThread;
	#endif
		NeuralNetProcessLifetimeManager()=delete;
		NeuralNetProcessLifetimeManager(NeuralNet* parent);
		NeuralNetProcessLifetimeManager (const NeuralNetProcessLifetimeManager&) = delete;
		NeuralNetProcessLifetimeManager(NeuralNetProcessLifetimeManager&&) = delete;
		NeuralNetProcessLifetimeManager& operator=(const NeuralNetProcessLifetimeManager&) = delete;
		NeuralNetProcessLifetimeManager& operator=(NeuralNetProcessLifetimeManager&&) noexcept = delete;
		~NeuralNetProcessLifetimeManager();
	};
	NeuralNetProcessLifetimeManager process;

protected:
	friend class Layer;
	friend class NeuralNetTest;
#if (DEBUG_NN_STATE_RECORDER)
	NNStateRecorder recorder;
#endif
	bool layerNorm;

	///////////CONTROLS ACTIVATION FUNCTION///////////
	ActivationFunction af = ActivationFunction::Sigmoid; //default mode

	double (*ActFunc)(const double&);
	double (*ActFuncDeriv)(const double&);

	//////////////////////////////////////////////////

	///////////CONTROLS COST FUNCTION/////////////////
	CostFunction cf = CostFunction::Quadratic;           //default mode

	double (*CostFunc)(const Matrix&, const Matrix&);
	Matrix (*CostFuncPartialDeriv)(const Matrix&, const Matrix&);

	template<class T>
	static inline double* convertToDouble(char* c) { return reinterpret_cast<double*>(c); }

	//////////////////////////////////////////////////
	

	static uint32_t fileVersioner(HeaderStatus& status, HeaderBuffer_t& header);

	//there is no access locking in these routines, they are internal and
	//rely on the calling code to make sure it is safe to access *this
	void headerSetupReadCallback(HeaderStatus& status, HeaderBuffer_t& header);
	void headerSetupWriteCallback(HeaderStatus& status, HeaderBuffer_t& header) const;
public:
	//These constants create boundaries for the program so the user can't jack things up
	const size_t	AP_MinLayerSize = 1;
	const size_t	AP_MaxLayerSize = 2000;
	const size_t	AP_MinNumLayer = 1;
	const size_t	AP_MaxNumLayer = 12;
	const double	AP_MinEta = 0.000000000000000000000000001;
	const double	AP_MaxEta = 200000;
	const size_t	AP_MinBatchSize = 1;
	const size_t	AP_MaxBatchSize = 60000;

	static double quadratic(const Matrix&, const Matrix&);
	static Matrix quadratic_partialderiv(const Matrix&, const Matrix&);

	static double sigmoid(const double& x);
	static double ReLU(const double& x);
	static double sigmoid_deriv(const double& x);
	static double ReLU_deriv(const double& x);

	NeuralNet() = delete;
	NeuralNet(const NeuralNetParams_t&);
	NeuralNet(const string&, NeuralNetParams_t&);
	NeuralNet (const NeuralNet&) = delete;
	NeuralNet(NeuralNet&&) = delete;
	NeuralNet& operator=(const NeuralNet&) = delete;
	NeuralNet& operator=(NeuralNet&&) noexcept = delete;

	void verifyFiles();

	string getFileName() const; 
	void setFileName(const string&);
	const Matrix& getInputLayer() const;
	void getInputLayerDims(int& width, int& height) const;
	const std::vector<Layer>& getLayers() const;
	const Matrix& getOutputActivations(int& width, int& height) const;
	const Matrix& getOutputCostGrad(int& width, int& height) const;
	const Matrix& getDesiredOutputs(int& width, int& height) const;
	double getCost() const;
	size_t getCompletedBatches() const; 
	size_t getBatchImgSize() const; 
	size_t getBatchLblSize() const;
	static std::filesystem::path testingRecordFilename();
	static std::filesystem::path trainingRecordFilename();
	void enableLayerNorm();
	void disableLayerNorm();
	void setActFunction(ActivationFunction);
	void setCostFunction(CostFunction);
	double calcActFunc(const double&);
	double calcActFuncDeriv(const double&);
	double calcCostFunc(const double&);

	void setRandDInputs();
	void setRandWBAcrossNetwork();
	void setRandDOutputs();
	void setTrainingParams(const size_t&, const int&, const double& eta);

	void updateNetwork();
	void computeBackpropErrors();
	void addCost();
	void avgCost(const size_t&);

	void resetCost();
	void resetWBCostGrad();
	void addWBCosts();
	void avgWBCosts(const size_t&);
	void applyGrad();

	void printInputs() const;
	void printActivations() const;
	void printZActivations() const;
	void printLayers() const;
	void printOutputs() const;
	void printCost() const;
	void printOutputCostGrad() const;
	void printLyrCostGrad() const;
	void printWBCostGrad() const;

	void saveFile() const;
	void readFile();
    std::filesystem::path getRecordRootPath() const;
	
	#ifdef UNIT_TEST_CONTEXT
	static void g_headerSetupWriteCallback(HeaderStatus& status, HeaderBuffer_t& header);
	#endif 

	void setRecordingState(bool record);
	bool setupNextTrainingBatch(bool shuffle=true);
	bool setupNextTestingBatch(bool shuffle=true);
	void performNextTrainingIteration(const size_t&, const std::vector<double>&);
	void performNextTestingIteration(const size_t&, const std::vector<double>&);
	void training();
	bool isProcessingHappening();
	bool hasProcessingStopBeenRequested();
	static void	trainingWorker(NeuralNet*, NeuralNetLabState_t&);
	static void	testingWorker(NeuralNet*, NeuralNetLabState_t&);
	void startTraining(NeuralNetLabState_t& nnls);
	void startTesting(NeuralNetLabState_t& nnls);
	void stopProcess(NeuralNet::ProcessingTypes);
	void userInteractiveTraining(NeuralNetLabState_t& nnls);
	void userInteractiveTesting(NeuralNetLabState_t& nnls);
	void trainingSetup(bool setRandomWB=true);
	void testingSetup();
	void training_TEST();
	double testing(int32_t);
	void testing_TEST();
};

enum class recordLoadState {None,Success,noRecordingsFound,couldntLoad};

struct NormData_t
{
    float min;
    float max;
};

struct NeuralLabViewer_t
{
	int											selectedImageType				= 0;
	int 										selectedLayer					= 0;
	string 										imageStr						= string("");
    NormData_t									normData 						= {};
	std::vector<float>							data 							= {};
	int											height							= 0;
	int											width							= 0;
	ProcessingRecordDisplayCfg_T				procRecordDispType				= ProcessingRecordDisplayCfg_T::inputImage;
#if (DEBUG_NN_STATE_RECORDER)				
	bool 										recordLoaded 					= false;
	bool 										anchorViewer 					= false;
	recordLoadState 							recordLoad 						= recordLoadState::None;
	std::vector<std::shared_ptr<NNStateRecord>> recordStore                     	= {};
	size_t										getRecordIdx();
	size_t 										recordIdx 						= 0;
#endif
};

struct etaPerformanceRecord_t
{
	double eta;
	double cost;
	double percentCorrect;
};

struct chunkPos
{
	bool valid = false;
	size_t chunkIdx = 0;
	std::streampos pos;
};

    // How many records we hold in testing/training results buffer
static constexpr size_t 						maxNumResults 							= 1000;
using resultsStorage_t = std::array<ProcessingRecord_t, maxNumResults>;

struct NeuralNetLabState_t {
public:
private:
    resultsStorage_t 	trainingResults							= {};
    resultsStorage_t 	testingResults							= {};
	size_t				trainingResultsIdx						= 0;
	size_t				testingResultsIdx						= 0;
	string				processingResultPath					= string("");

	void updateTrainingResultIdx(size_t idx);
	void updateTestingResultIdx(size_t idx);

public:
	void setProcessingResultPath(const string&);
	const string& getProcessingResultPath() const;
	const string& getAndCreateProcessingResultPath();

	void setCurrentTrainingResults(size_t, const ProcessingRecord_t&); 
	void setCurrentTestingResults(size_t, const ProcessingRecord_t&); 
	const ProcessingRecord_t& getCurrentTrainingResults() const; 
	const ProcessingRecord_t& getCurrentTestingResults() const; 
	const size_t& getCurrentTrainingResultsIdx() const; 
	const size_t& getCurrentTestingResultsIdx() const; 
	void incrTrainingResultIdx();	
	void decrTrainingResultIdx();	
	void resetTrainingResultsIdx();	
	void setTrainingResultsIdx(size_t);	
	void resetTrainingResults();	
	void incrTestingResultIdx();
	void decrTestingResultIdx();
	void resetTestingResultsIdx();
	void setTestingResultsIdx(size_t);
	void resetTestingResults();

public:
	enum class eChannelChoices : uint32_t {
		eCurrentState = 0,
#if (DEBUG_NN_STATE_RECORDER)
		recordingViewer,
#endif			
		eTrainingRecords, 
		eTestingRecords, 
	};

	enum class eStatViewerChoices : uint32_t {
		eNone = 0,
		eHistogram,
		eFFT,
		ePixelValue
	};
	enum class eWBRandomizationChoices : uint32_t {
		eNone = 0,
		eOnFirstEpoch,
		eOnEveryEpoch
	};

	static constexpr const char* 				channelChoices[] 						= {
		"Current State", 
#if (DEBUG_NN_STATE_RECORDER)
		"Matrix Records",
#endif			
		"Training Records", 
		"Testing Records", 
	};


	static constexpr const char* 					statViewerChoices[] 					= {"None", "Histogram", "FFT", "Pixel Value"};
	static constexpr const char* 					wbRandomizationChoices[] 				= {"None", "On 1st Epoch", "On Every Epoch"};

	static constexpr const char* 					outputActiveComparisonStr				= "outputActiveComparison";
	std::vector<const char*>						viewerSelectableDisplayStrs				= {};
	NeuralLabViewer_t								viewer1									= {};
	NeuralLabViewer_t								viewer2									= {};

	std::vector<string> 							layerstr 								= {};
	std::vector<const char*> 						layers 									= {};
    std::vector<const char*> 						selectableNNImage 						= {};
	bool 											processingInitialized 					= false;
	bool 											completeLastBatch 						= true;
	bool 											shuffleSets 							= true;
	bool 											randomizeWeightsBiases 					= true;
	int												wbRandomizationSelection				= (int)ECBR_EcTy(NeuralNetLabState_t::eWBRandomizationChoices, eOnFirstEpoch);
	bool 											wbHaveBeenRandomized 					= false;
	bool 											neuralNetLabTabReloadRequired 			= false;
	bool 											trainingTestingSectionReloadRequired 	= false;
	bool 											processingRunEntire 					= false;
	NeuralNet::ProcessingTypes 						lastProcessType 						= NeuralNet::ProcessingTypes::None;
	size_t 											batchStep 								= 0;
	size_t 											processingStep							= 0;
	size_t 											repeatCount								= 0;
	bool											etaResultFlush 							= false;
	bool											trainingResultFlush 					= false;
	bool											testingResultFlush 						= false;
	bool											trainingResultsLoaded 					= false;
	size_t											trainingChunkIdx						= 0;
	bool 											trainingChunkReloadNeeded				= false;
    size_t 											trainingResultsCount					= 0;
    std::vector<chunkPos>		 					trainingResultsFileChunkPos				= {};
	size_t											trainingBufferChunkIdx					= 0;
	bool											testingResultsLoaded 					= false;
	size_t											testingChunkIdx							= 0;
	bool 											testingChunkReloadNeeded				= false;
    size_t 											testingResultsCount						= 0;
    std::vector<chunkPos>		 					testingResultsFileChunkPos				= {};
    size_t											trainingChunkReloadIdx					= 0;
    size_t											testingChunkReloadIdx					= 0;

	bool											etaStepping 							= false;
	std::chrono::steady_clock::time_point			lastEtaStatusCheck						= {};
	bool											startTraining							= false;
	bool											parallelEtaProcessing					= false;
	bool											trainingInProgress						= false;
	bool											finishTraining  						= false;
	bool											startTesting							= false;
	bool											testingInProgress						= false;
	bool											finishTesting  							= false;
    double											originalEta								= 0.0;
    double											startingEta								= 0.0;
    double											endingEta								= 0.0;
    double											stepEta								    = 0.0;
    double											currentEta								= 0.0;
	std::vector<etaPerformanceRecord_t>				etaRecords								= {};
	std::vector<string>								etaRecordsDisplay						= {};
	bool											etaLogStep 								= false;

	double 											currentCost 							= 0.0;
	double 											currentPercentCorrect 					= 0.0;
	size_t 											epochsRemaining							= 0;

	int 											channelSelect 							= 0;
	int 											statViewerSelect 						= 0;
#if (DEBUG_NN_STATE_RECORDER)				
	bool 											record 									= false;
	string 											recordingPath 							= string("");
#endif	
	bool 											playback								= false;
	float 											playbackSpeed							= 1.00f;
	bool											reload									= false;
	bool											recordReset								= false;


	
};

