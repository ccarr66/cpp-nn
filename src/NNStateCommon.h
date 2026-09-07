#pragma once
#include "GLOBALS.h"


enum class nnStateType : uint32_t
{
	eInvalidTypeLow	= 0				, //= 0,
    eNonLayer_DesiredOutputs		, //= BIT(0),
    eNonLayer_InputLayer			, //= BIT(1),
    eNonLayer_OutputCostGrad		, //= BIT(2),
    eNonLayer_WBCostGrad			, //= BIT(3),
    eMaxNonLayer                    , //= eNonLayer_WBCostGrad,
    eLayer_prevLayerActivations		, //= BIT(4),
    eLayer_Z_Activations			, //= BIT(5),
    eLayer_Activations				, //= BIT(6),
    eLayer_Weights					, //= BIT(7),
    eLayer_Biases					, //= BIT(8),
    eLayer_BiasCosts				, //= BIT(9),
    eLayer_WeightCosts				, //= BIT(10),
	eInvalidTypeHigh                , //= BIT(11),
};

struct nnDisplayStr {
	const char* str; bool print;
};

static constexpr nnDisplayStr nnTypeDisplayStrs[] = {
    {"eInvalidTypeLow",            false},
    {"NonLayer_DesiredOutputs",    true},
    {"NonLayer_InputLayer",        true},
    {"NonLayer_OutputCostGrad",    true},
    {"NonLayer_WBCostGrad",        true},
    {"eMaxNonLayer",               false},
    {"Layer_prevLayerActivations", true},
    {"Layer_Z_Activations",        true},
    {"Layer_Activations",          true},
    {"Layer_Weights",              true},
    {"Layer_Biases",               true},
    {"Layer_BiasCosts",            true},
    {"Layer_WeightCosts",          true},
    {"eInvalidTypeHigh",           false},
};
bool isValidNNStateType(nnStateType type);
nnStateType getNNTypeFromStr(string type);
string getNNTypeStr(nnStateType type);
#define isLayerType(type) ( ECBR_ValTy( nnStateType, type ) > ECBR_EcTy( nnStateType, eMaxNonLayer ) )