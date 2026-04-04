#include "NNStateCommon.h"
#include "ErrorHandler.h"

bool isValidNNStateType(nnStateType type)
{
    bool result = false;
	switch (type) 
	{
        case nnStateType::eNonLayer_DesiredOutputs:
        case nnStateType::eNonLayer_InputLayer:
        case nnStateType::eNonLayer_OutputCostGrad:
        case nnStateType::eNonLayer_WBCostGrad:
        case nnStateType::eLayer_prevLayerActivations:
        case nnStateType::eLayer_Z_Activations:
        case nnStateType::eLayer_Activations:
        case nnStateType::eLayer_Weights:
        case nnStateType::eLayer_Biases:
        case nnStateType::eLayer_BiasCosts:
        case nnStateType::eLayer_WeightCosts:
            result = true;
            break;
        default:
            break;
	}
    return result;
}

static constexpr const char* nnTypeStrs[] = {
"eInvalidTypeLow",
"NonLayer_DesiredOutputs",
"NonLayer_InputLayer",
"NonLayer_OutputCostGrad",
"NonLayer_WBCostGrad",
"eMaxNonLayer",
"Layer_prevLayerActivations",
"Layer_Z_Activations",
"Layer_Activations",
"Layer_Weights",
"Layer_Biases",
"Layer_BiasCosts",
"Layer_WeightCosts",
"eInvalidTypeHigh"
};

nnStateType getNNTypeFromStr(string type)
{
    nnStateType nnType = nnStateType::eInvalidTypeLow;
    size_t nnTypeIdx = 0;
    for (const auto str : nnTypeStrs)
    {
        if (type == string(str))
            nnType = static_cast<nnStateType>(nnTypeIdx);
        nnTypeIdx++;
    }
    return nnType;
}

string getNNTypeStr(nnStateType type)
{
	string result = string("UNKNOWN");
	switch (type)
	{
        case nnStateType::eNonLayer_DesiredOutputs:
            result=string("NonLayer_DesiredOutputs");
            break;
        case nnStateType::eNonLayer_InputLayer:
            result=string("NonLayer_InputLayer");
            break;
        case nnStateType::eNonLayer_OutputCostGrad:
            result=string("NonLayer_OutputCostGrad");
            break;
        case nnStateType::eNonLayer_WBCostGrad:
            result=string("NonLayer_WBCostGrad");
            break;
        case nnStateType::eLayer_prevLayerActivations:
            result=string("Layer_prevLayerActivations");
            break;
        case nnStateType::eLayer_Z_Activations:
            result=string("Layer_Z_Activations");
            break;
        case nnStateType::eLayer_Activations:
            result=string("Layer_Activations");
            break;
        case nnStateType::eLayer_Weights:
            result=string("Layer_Weights");
            break;
        case nnStateType::eLayer_Biases:
            result=string("Layer_Biases");
            break;
        case nnStateType::eLayer_BiasCosts:
            result=string("Layer_BiasCosts");
            break;
        case nnStateType::eLayer_WeightCosts:
            result=string("Layer_WeightCosts");
            break;
        default:
            ErrorHandler::FatalError(string("Invalid NN Type!"));
            break;
	}
	return result;
}
