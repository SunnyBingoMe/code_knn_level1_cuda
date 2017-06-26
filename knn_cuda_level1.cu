/* must have defines */
#define TIMER_OK 1
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

/* general include */
#include "wb.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "sunnyF.h"
#include "sunnyCudaF.h"
#include <thrust/scan.h>

/* project spedific include */
#include "cuda_gpu_r_knn.h"
#include <stdio.h>
#include <stdlib.h>
#include "mat.h"
#include "mex.h"
#include <iostream>
#include <thrust/sort.h>
using namespace std;

/* project specific define */
#define Stage                   1       // 1: weight gen / train. 2: predict / test. 3: analysis flag. 4: re-norm weights
#define Macro_id                1
#define Data_variable_nr        2
#define Daily_record_nr         288
#define Daily_data_nr           576     // 288 * 2variables
#define Block_size              288
#define Block_size_2            256
#define Error_is_rmse           0
#define OneDay_multiUsage       1
#define TrainingDataRatio       0.8f
#define Max_searchStepLength    256
#define Max_matchNPredictRange  329     // minimum requirement: 329 = 256 + 32 + 1 + 32 + 8 ; but 288 * 3 is easier to handle. // each block-shared-mem can store max 21 days two-variable float data.
#define Distance_type           'E'
#define Default_distance_of_day 9999.0f
#define IncidentToleranceRatio  0.1f
#define NormalizeDiscardRatio   0.75f   // worst will be scored as 0 at the final round, -1 means do nothing. (unit is 1). OBS: not considering value-frequency.
#define Score_valueMethod       0       /* 0: default rank-linear subtraction.
                                           1: dist reciprocal linear (pow_1).
                                           2: dist reciprocal pow_2.
                                           3: dist subtraction linear (pow_1).
                                           4: dist subtraction pow_2.
                                           5: exponential of dist.
                                           6: exponential of rank.
                                        */
#define Score_minDistanceForDivision    0.001f
#define Score_maxDistanceForSubtraction 50.00f
#define Score_RankingPunishMultiply     2.0f    // pre-requirement: Score_valueMethod == 0.
#define Score_RankingPunishNrRatio      0.0f    // pre-requirement: Score_valueMethod == 0. worst will be scored as 0 in each ranking round. (unit is 1, e.g. 0.03 = 3%)
#define Score_regardsE2d                0
#define Score_useFlowSegment        0   // will use only the 1st segment
#define DailyUpDown_segment_nr      2
#define FlowIndicator_segment_nr    10
#define Remove_incident_neighbor    1
#if Stage == 1
#define Write_ph_output         0
#define Write_analysisFlag      0
#define Do_kernel               1
#define WeightExists_isTesting  0
#define NormalizeWeightOnly     0
#elif Stage == 2
#define Write_ph_output         0
#define Write_analysisFlag      1
#define Do_kernel               1
#define WeightExists_isTesting  1
#define NormalizeWeightOnly     0
#elif Stage == 3
#define Write_ph_output         0
#define Write_analysisFlag      1
#define Do_kernel               0
#define WeightExists_isTesting  1
#define NormalizeWeightOnly     0
#elif Stage == 4
#define Write_ph_output         0
#define Write_analysisFlag      0
#define Do_kernel               0
#define WeightExists_isTesting  1
#define NormalizeWeightOnly     1       // pre-requirement: WeightExists_isTesting && !Do_kernel
#endif

/* debug (general & project specific) */
#define Debug_queryDayIndex         1
#define Debug_logic                 0
#define Debug_show_param            0
#define Debug_recordOffset_ofDay    0
#define Debug_detail_timer          0   // OBS warn: if no screen detected, terminal output will cause program to pause.
#define Debug_allDayDistance        0
#define Debug_sortedKnnPredict      0   // pre-requirement: Debug_allDayDistance
#define Debug_windowSize            0
#define Debug_write_ph_output_ratio 1   // pre-requirement: Write_ph_output
#define Debug_incidentFlagListing_queryData    0
#define Debug_analysisFlagListing_queryData    0
#define Debug_weightOfCombinationNIndex_txt    0
#define Debug_weightOfCombination              0
#define Debug_kernel4_normalize                0
#define Debug_weightedPredictDistance          0    // pre-requirement: WeightExists_isTesting
#define Debug_queryRecord_flowIndicator        0

/* general macro-IDs defines */
#if Macro_id
#define localId_col  threadIdx.x
#define localId_row  threadIdx.y
#define localId_page threadIdx.z
#define localId_1D   localId_col
#define localId_2D   (localId_row * Block_width + localId_col)
#define localId_3D   ((localId_col*Block_height + localId_row)*Block_pages + localId_page)
#define localId      localId_3D
#define blockId_col  blockIdx.x
#define blockId_row  blockIdx.y
#define blockId_page blockIdx.z
#define blockId_1D   blockId_col
#define blockId_2D   (blockId_row * Grid_width + blockId_col)
#define blockId_3D   ((blockId_col*Grid_height + blockId_row)*Grid_pages + blockId_page)
#define blockId      blockId_3D
#define globalId_col (blockId_col * Block_width + localId_col)
#define globalId_row (blockId_row * Block_height + localId_row)
#define globalId_1D  globalId_col
#define globalId_2D  (globalId_row * Block_width*Grid_width + globalId_col)
#define globalId_3D  blockId_3D * OneBlock_size3D + localId_3D
#define globalId     globalId_3D
#endif

/* usage:
*/


/* =============================================================== kernel 1: param win & h =============================================================== */
/* =============================================================== kernel 1: param win & h =============================================================== */
__global__ void calculateOneTimePoint_searchShiftNSearchStepLength(
    int    queryDayIndex_inAllGlobalDays, int recordOffset_ofDay,
#if Write_ph_output
    float *pd_output,
#endif
    float *pd_oneTimePoint_searchStepLength_searchShift_day_distance,
    float *pd_input, int inputRecordTotalNr, int completeDayNr, int searchableDayNr,
    int   *pd_incidentFlagScannedListing,
    int   *pd_searchStepLengthListing, int searchStepLengthNr,
    int   *pd_windowSizeListing, int windowSizeNr,
    int   *pd_kLevel1Listing, int kLevel1Nr,
    int   *pd_predictStepLengthListing, int predictStepLengthNr)
{
    // IDs
#if !Macro_id
    int localId_col   = threadIdx.x;
    int localId_row   = threadIdx.y;
    int localId_1D    = localId_col;
    int localId_2D    = (localId_row * Block_width + localId_col);
    int blockId_col   = blockIdx.x;
    int blockId_row   = blockIdx.y;
    int blockId_1D    = blockId_col;
    int blockId_2D    = (blockId_row * Grid_width + blockId_col);
    int globalId_col  = (blockId_col * Block_width + localId_col);
    int globalId_row  = (blockId_row * Block_height + localId_row);
    int globalId_1D   = globalId_col;
    int globalId_2D   = (globalId_row * Block_width*Grid_width + globalId_col);
#endif
    __shared__ float pb_queryData[Daily_data_nr * 3];
    __shared__ float pb_historyData[Daily_data_nr * 3];
    __shared__ float pb_scanedDistance[Max_searchStepLength];
#if Remove_incident_neighbor
    __shared__ int   pb_incidentFlagScannedListing[Daily_record_nr * 3];
    __shared__ int   pb_incidentNrDiffCeilingOfHistoryListing[Max_searchStepLength];
    __shared__ int    b_incidentFlagScanned_ceilingOfHistory_index;
    __shared__ int    b_incidentFlagScanned_ceilingOfPredict_index;
#endif

    int searchStepLength_max = pd_searchStepLengthListing[searchStepLengthNr - 1];
    int predictStepLength_max = pd_predictStepLengthListing[predictStepLengthNr - 1];
    int copyStartOffsetIndexGlobal;
    int perThread_dataFromGpuRamIndex;
    int perBlock_dayIndex_in_analysedDays = blockId_col;
#if Remove_incident_neighbor
    int perThread_pb_incidentFlagScannedIndex;
#endif

    // copy query data (3 days) to shared mem, using max searchStepLengthIndex //TODO: improve: flexible regarding block-size
    copyStartOffsetIndexGlobal = (queryDayIndex_inAllGlobalDays - 1) * Daily_data_nr;
    perThread_dataFromGpuRamIndex = copyStartOffsetIndexGlobal + localId_1D;
    for (int i = 0; i < 3 * Data_variable_nr; i++){
        pb_queryData[localId_1D + 288 * i] = pd_input[perThread_dataFromGpuRamIndex + 288 * i];
    }

    // copy history data (3 days search & predict) to shared mem, using max-search &  max-predict & max-shift //TODO: improve: flexible regarding block-size
    copyStartOffsetIndexGlobal = perBlock_dayIndex_in_analysedDays * Daily_data_nr;
    perThread_dataFromGpuRamIndex = copyStartOffsetIndexGlobal + localId_1D;
    for (int i = 0; i < 3 * Data_variable_nr; i++){
        pb_historyData[localId_1D + 288 * i] = pd_input[perThread_dataFromGpuRamIndex + 288 * i];
    }

#if Remove_incident_neighbor
    // copy incident flag (3 days), using max-search &  max-predict & max-shift //TODO: improve: flexible regarding block-size
    copyStartOffsetIndexGlobal = perBlock_dayIndex_in_analysedDays * Daily_record_nr;
    perThread_dataFromGpuRamIndex = copyStartOffsetIndexGlobal + localId_1D;
    for (int i = 0; i < 3; i++){
        pb_incidentFlagScannedListing[localId_1D + 288 * i] = pd_incidentFlagScannedListing[perThread_dataFromGpuRamIndex + 288 * i];
    }
#endif

    __syncthreads();
    // do calculation
    int windowSize_max = pd_windowSizeListing[windowSizeNr - 1];
    int searchShiftNr_max = windowSize_max * 2 + 1;
    for (int searchShift_inMaxWindow = -windowSize_max; searchShift_inMaxWindow <= windowSize_max; searchShift_inMaxWindow++){
        //TODO: improve: use max windowSizeIndex and cache different combination with searchStepLengthIndex

        // calculate distance to pb_scanedDistance, using max searchStepLengthIndex
        if (localId_1D < searchStepLength_max){
            int perThread_pb_queryDataIndex   = (Daily_record_nr + recordOffset_ofDay - localId_1D - 1) * Data_variable_nr;
            int perThread_pb_historyDataIndex = perThread_pb_queryDataIndex + searchShift_inMaxWindow * Data_variable_nr;
#if Remove_incident_neighbor
            perThread_pb_incidentFlagScannedIndex = perThread_pb_historyDataIndex / Data_variable_nr;
            if (localId_1D == 0){
                b_incidentFlagScanned_ceilingOfHistory_index = perThread_pb_incidentFlagScannedIndex;
                b_incidentFlagScanned_ceilingOfPredict_index = perThread_pb_incidentFlagScannedIndex + predictStepLength_max;
            }
            perThread_pb_incidentFlagScannedIndex -= 1;
            perThread_pb_incidentFlagScannedIndex = max(perThread_pb_incidentFlagScannedIndex, 0);
#endif
#if Distance_type == 'E'
            // calculate Euclidean distance
            float t_rate_error  = pb_queryData[perThread_pb_queryDataIndex] - pb_historyData[perThread_pb_historyDataIndex];
            float t_speed_error = pb_queryData[perThread_pb_queryDataIndex + 1] - pb_historyData[perThread_pb_historyDataIndex + 1];
            pb_scanedDistance[localId_1D] = sqrt(t_rate_error * t_rate_error + t_speed_error * t_speed_error);
#else
            static assert(false, "Unknown Neighbour Distance Type");
#endif
        }

        // per-block scan to shared pb_scanedDistance, using max searchStepLengthIndex
        __syncthreads();
        int stride, perThread_index;
        // scan step 1 -> reduce
        for (stride = 1; stride <= 128; stride *= 2){
            __syncthreads();
            if (localId_1D < 128){
                perThread_index = (localId_1D + 1) * (stride * 2) - 1;
                if (perThread_index < 256){
                    pb_scanedDistance[perThread_index] += pb_scanedDistance[perThread_index - stride];
                }
            }
        }
        // scan step 2 -> reverse reduce
        __syncthreads();
        for (stride = 128 / 2; stride >= 1; stride /= 2){
            __syncthreads();
            if (localId_1D < 128){
                perThread_index = (localId_1D + 1) * (stride * 2) - 1;
                if (perThread_index + stride < 256){
                    pb_scanedDistance[perThread_index + stride] += pb_scanedDistance[perThread_index];
                }
            }
        }

        // division, due to the distance is sum result.
        __syncthreads();
        if (localId_1D < Max_searchStepLength){
            pb_scanedDistance[localId_1D] /= (float)(localId_1D + 1);
        }

#if Remove_incident_neighbor
        // per search-h incident nr
        __syncthreads();
        if (localId_1D < searchStepLength_max){
            pb_incidentNrDiffCeilingOfHistoryListing[localId_1D] = pb_incidentFlagScannedListing[b_incidentFlagScanned_ceilingOfHistory_index] - pb_incidentFlagScannedListing[perThread_pb_incidentFlagScannedIndex];
        }
#endif

        // for each h ; then write to global mem
        __syncthreads();
        if (localId_1D < searchStepLengthNr){
            int searchShiftIndex_inMaxWindow;
            searchShiftIndex_inMaxWindow = searchShift_inMaxWindow + windowSize_max;
            int perThread_searchStepLengthIndex = localId_1D;
            int perThread_searchStepLength = pd_searchStepLengthListing[perThread_searchStepLengthIndex];

            // get 3D index: [perThread_searchStepLengthIndex][searchShiftIndex_inMaxWindow][dayIndex_in_analysedDays == blockId_1D]
            int perThread_indexIn_searchStepLength_searchShift_day = (perThread_searchStepLengthIndex * searchShiftNr_max + searchShiftIndex_inMaxWindow) * searchableDayNr + blockId_1D;
            int queryDayIndex_in_analysedDays = queryDayIndex_inAllGlobalDays - 1;
            if (queryDayIndex_in_analysedDays != perBlock_dayIndex_in_analysedDays){
                pd_oneTimePoint_searchStepLength_searchShift_day_distance[perThread_indexIn_searchStepLength_searchShift_day] = pb_scanedDistance[perThread_searchStepLength - 1];
            }
            else{
                pd_oneTimePoint_searchStepLength_searchShift_day_distance[perThread_indexIn_searchStepLength_searchShift_day] = Default_distance_of_day;
            }
#if Remove_incident_neighbor
            int incidentNr_inHistory = pb_incidentNrDiffCeilingOfHistoryListing[perThread_searchStepLength - 1];
            int incidentNr_inPredict = pb_incidentFlagScannedListing[b_incidentFlagScanned_ceilingOfPredict_index] - pb_incidentFlagScannedListing[b_incidentFlagScanned_ceilingOfHistory_index];
            if (incidentNr_inHistory > perThread_searchStepLength * IncidentToleranceRatio || incidentNr_inPredict > 0){
                pd_oneTimePoint_searchStepLength_searchShift_day_distance[perThread_indexIn_searchStepLength_searchShift_day] = Default_distance_of_day;
            }
#endif
        }
        __syncthreads();
    }/* for each shift in window end */
    __syncthreads();
}


/* =============================================================== kernel 2: param k & pred h =============================================================== */
/* =============================================================== kernel 2: param k & pred h =============================================================== */
__global__ void init_pd_oneTimePoint_searchStepLength_searchShift_day_index(
    int *pd_oneTimePoint_searchStepLength_searchShift_day_index, 
    int  oneTimePointAllNeighborDistanceNr)
{
    if (globalId_2D < oneTimePointAllNeighborDistanceNr){
        pd_oneTimePoint_searchStepLength_searchShift_day_index[globalId_2D] = globalId_2D;
    }
}

__global__ void calculateOneTimePoint_kLevel1NPredictStepLength(
#if Debug_sortedKnnPredict
    float *pd_sortedKnnPrediction,
#endif
#if Debug_weightedPredictDistance
    float *pd_unweightedPredictListing,
#endif
#if WeightExists_isTesting
    float *pd_weightListing, int flowIsGoingDown, int flowIndicatorIndex, float *pd_allCombination_weightedPredictListing,
#elif !WeightExists_isTesting
    int    queryRecordOkForWeight,
#endif
    int    searchStepLengthIndex, int windowSizeIndex,
    float *pd_perTimePoint_combinationMeasurementListing,
    int    queryDayIndex_inAllGlobalDays, int recordOffset_ofDay,
#if Write_ph_output
    float *pd_output,
#endif
    float *pd_oneTimePoint_searchStepLength_searchShift_day_distance,
    int   *pd_oneTimePoint_searchStepLength_searchShift_day_index,
    float *pd_input, int inputRecordTotalNr, int completeDayNr, int searchableDayNr, 
    int   *pd_searchStepLengthListing, int searchStepLengthNr,
    int   *pd_windowSizeListing, int windowSizeNr,
    int   *pd_kLevel1Listing, int kLevel1Nr,
    int   *pd_predictStepLengthListing, int predictStepLengthNr)
{
    // IDs
#if !Macro_id
    int localId_col   = threadIdx.x;
    int localId_row   = threadIdx.y;
    int localId_1D    = localId_col;
    int localId_2D    = (localId_row * Block_width + localId_col);
    int blockId_col   = blockIdx.x;
    int blockId_row   = blockIdx.y;
    int blockId_1D    = blockId_col;
    int blockId_2D    = (blockId_row * Grid_width + blockId_col);
    int globalId_col  = (blockId_col * Block_width + localId_col);
    int globalId_row  = (blockId_row * Block_height + localId_row);
    int globalId_1D   = globalId_col;
    int globalId_2D   = (globalId_row * Block_width*Grid_width + globalId_col);
#endif

    int windowSize = pd_windowSizeListing[windowSizeIndex];
    int windowSize_max = pd_windowSizeListing[windowSizeNr - 1];
    int searchShiftNr_max = windowSize_max * 2 + 1;

    __shared__ float pb_sortedDistance_asKey[256 * 2];   // === OneBlock_size2D * 2, but need const value
    __shared__ int pb_sortedConfigIndex_asValue[256 * 2];

    // memset
    pb_sortedDistance_asKey[localId_2D] = Default_distance_of_day;
    pb_sortedDistance_asKey[localId_2D + OneBlock_size2D] = Default_distance_of_day;
    pb_sortedConfigIndex_asValue[localId_2D] = -1;
    pb_sortedConfigIndex_asValue[localId_2D + OneBlock_size2D] = -1;

    // copy pd_oneTimePoint_searchStepLength_searchShift_day_distance to pb_oneTimePoint_searchShift_day_distance // 3D index: [searchStepLengthIndex][searchShiftIndex_inMaxWindow][dayIndex_in_analysedDays]
    __syncthreads();
    int searchShiftIndex_inMaxWindow_left = -windowSize + windowSize_max;
    int searchShiftIndex_inMaxWindow_right = windowSize + windowSize_max;
    int dataToCopy_oneTimePointDistance_index_left  = (searchStepLengthIndex * searchShiftNr_max + searchShiftIndex_inMaxWindow_left)  * searchableDayNr + 0; // [searchStepLengthIndex][left] [dayIndex_in_analysedDays == 0]
    int dataToCopy_oneTimePointDistance_index_right = (searchStepLengthIndex * searchShiftNr_max + searchShiftIndex_inMaxWindow_right) * searchableDayNr + (searchableDayNr - 1); // [searchStepLengthIndex][right][dayIndex_in_analysedDays == last]
    int perThread_indexIn_searchStepLength_searchShift_day_distance = dataToCopy_oneTimePointDistance_index_left + localId_2D;
    if (perThread_indexIn_searchStepLength_searchShift_day_distance <= dataToCopy_oneTimePointDistance_index_right){
        pb_sortedDistance_asKey[localId_2D] = pd_oneTimePoint_searchStepLength_searchShift_day_distance[perThread_indexIn_searchStepLength_searchShift_day_distance];
        pb_sortedConfigIndex_asValue[localId_2D] = pd_oneTimePoint_searchStepLength_searchShift_day_index[perThread_indexIn_searchStepLength_searchShift_day_distance];
    }
    perThread_indexIn_searchStepLength_searchShift_day_distance += OneBlock_size2D;
    if (perThread_indexIn_searchStepLength_searchShift_day_distance <= dataToCopy_oneTimePointDistance_index_right){
        pb_sortedDistance_asKey[localId_2D + OneBlock_size2D] = pd_oneTimePoint_searchStepLength_searchShift_day_distance[perThread_indexIn_searchStepLength_searchShift_day_distance];
        pb_sortedConfigIndex_asValue[localId_2D + OneBlock_size2D] = pd_oneTimePoint_searchStepLength_searchShift_day_index[perThread_indexIn_searchStepLength_searchShift_day_distance];
    }
    
    /*
#if Debug_sortedKnnPredict
    __syncthreads();
    pd_sortedKnnPrediction[blockId_2D * Block_size_2*2 + localId_2D] = pb_sortedDistance_asKey[localId_2D];
    pd_sortedKnnPrediction[blockId_2D * Block_size_2*2 + localId_2D + 256] = pb_sortedDistance_asKey[localId_2D + 256];
#endif
    */

    /***** predict for each predictStepLength *****/
    __syncthreads();
    for (int predictStepLengthIndex = 0; predictStepLengthIndex < predictStepLengthNr; predictStepLengthIndex++){
        int predictStepLength = pd_predictStepLengthListing[predictStepLengthIndex];

        // copy related prediction to pb_, consider block as 1D; re-using the memory of pb_sortedDistance_asKey as prediction (all the mem: 256 * 2, as there are two variables)
        __syncthreads();
        int perThread_dataFromGpuRamIndex;
        perThread_dataFromGpuRamIndex = pb_sortedConfigIndex_asValue[localId_2D];
        int perThread_searchedDayIndex = perThread_dataFromGpuRamIndex % searchableDayNr;
        int perThread_searchShiftIndex_inMaxWindow = (perThread_dataFromGpuRamIndex / searchableDayNr) % searchShiftNr_max;
        int perThread_pd_predictStartDataIndexIn_historyData = ((perThread_searchedDayIndex + 1) * Daily_record_nr + recordOffset_ofDay + (perThread_searchShiftIndex_inMaxWindow - windowSize_max) + (predictStepLength - 1)) * Data_variable_nr;
        pb_sortedDistance_asKey[localId_2D] = pd_input[perThread_pd_predictStartDataIndexIn_historyData];                       // copy rate
        pb_sortedDistance_asKey[localId_2D + OneBlock_size2D] = pd_input[perThread_pd_predictStartDataIndexIn_historyData + 1]; // copy speed
        
#if Debug_sortedKnnPredict
        __syncthreads();
        pd_sortedKnnPrediction[blockId_2D * Block_size_2*2 + localId_2D] = pb_sortedDistance_asKey[localId_2D];
        pd_sortedKnnPrediction[blockId_2D * Block_size_2*2 + localId_2D + 256] = pb_sortedDistance_asKey[localId_2D + 256];
#endif

        // (weighted) two-variable-scan, consider block as 1D (localid)
        __syncthreads();
        int stride, perThread_index;
        // two-variable-scan step 1 -> reduce
        for (stride = 1; stride <= 128; stride *= 2){
            __syncthreads();
            perThread_index = (localId_2D % 128 + 1) * (stride * 2) - 1;
            if (perThread_index < 256){
                if (localId_2D < 128){
                    pb_sortedDistance_asKey[perThread_index] += pb_sortedDistance_asKey[perThread_index - stride];              // rate
                }
                else {
                    pb_sortedDistance_asKey[perThread_index + 256] += pb_sortedDistance_asKey[perThread_index - stride + 256];  // speed
                }
            }
        }
        // two-variable-scan step 2 -> reverse reduce
        __syncthreads();
        for (stride = 128 / 2; stride >= 1; stride /= 2){
            __syncthreads();
            perThread_index = (localId_2D % 128 + 1) * (stride * 2) - 1;
            if (perThread_index + stride < 256){
                if (localId_2D < 128){
                    pb_sortedDistance_asKey[perThread_index + stride] += pb_sortedDistance_asKey[perThread_index];              // rate
                }
                else {
                    pb_sortedDistance_asKey[perThread_index + stride + 256] += pb_sortedDistance_asKey[perThread_index + 256];  // speed
                }
            }
        }

        // division, due to the distance is sum result.
        // division is done when manadatory (5 lines later).

        /*
#if Debug_sortedKnnPredict
        __syncthreads();
        pd_sortedKnnPrediction[blockId_2D * Block_size_2*2 + localId_2D] = pb_sortedDistance_asKey[localId_2D];
        pd_sortedKnnPrediction[blockId_2D * Block_size_2*2 + localId_2D + 256] = pb_sortedDistance_asKey[localId_2D + 256];
#endif
        */

        /***** predict, consider block as 1D *****/
        __syncthreads();
        if (localId_2D < kLevel1Nr){
            int perThread_kLevel1Index = localId_2D;
            int perThread_kLevel1 = pd_kLevel1Listing[perThread_kLevel1Index];

            // (weighted) average/mean as prediction
            float perThread_ratePredict  = pb_sortedDistance_asKey[perThread_kLevel1 - 1] / (float)perThread_kLevel1;
            float perThread_speedPredict = pb_sortedDistance_asKey[perThread_kLevel1 - 1 + 256] / (float)perThread_kLevel1;
            //TODO: optional: deviation-option-1: save nn-prediction to calculate deviation/sigma, then the division above should be done earlier.

            // measure
#if Write_ph_output || !WeightExists_isTesting
            int predictStartDataIndex = (queryDayIndex_inAllGlobalDays * Daily_record_nr + recordOffset_ofDay + (predictStepLength - 1)) * Data_variable_nr;
            float rateReal  = pd_input[predictStartDataIndex];
            float speedReal = pd_input[predictStartDataIndex + 1];
#if Error_is_rmse
            float rateError = abs(perThread_ratePredict - rateReal);
            float speedError = abs(perThread_speedPredict - speedReal);
#elif !Error_is_rmse
            float rateError = perThread_ratePredict - rateReal;
            float speedError = perThread_speedPredict - speedReal;
#endif
#endif

            int tIndex;
#if Write_ph_output
            // output index (pred-aligned), for prediction: [queryDayIndex_inAllGlobalDays * Daily_record_nr + recordOffset_ofDay + predictStepLength - 1][searchStepLengthIndex][windowSizeIndex][perThread_kLevel1Index][predictStepLengthIndex][rateOrSpeed]
            tIndex = (((((queryDayIndex_inAllGlobalDays * Daily_record_nr + recordOffset_ofDay + predictStepLength - 1)*searchStepLengthNr + searchStepLengthIndex)*windowSizeNr + windowSizeIndex)*kLevel1Nr + perThread_kLevel1Index)*predictStepLengthNr + predictStepLengthIndex)* Data_variable_nr + 0;

            pd_output[tIndex] = rateError;
            pd_output[tIndex + 1] = speedError;
#endif
#if WeightExists_isTesting
            //TODO: optional: deviation-option-2: save nn-prediction to calculate deviation/sigma
            // get weight. index: [flowIsGoingDown][flowIndicatorIndex][pred-h][search-h][win][k1]
            tIndex = ((((flowIsGoingDown*FlowIndicator_segment_nr + flowIndicatorIndex)*predictStepLengthNr + predictStepLengthIndex)*searchStepLengthNr + searchStepLengthIndex)*windowSizeNr + windowSizeIndex)*kLevel1Nr + perThread_kLevel1Index;
            float weight_t = pd_weightListing[tIndex];
            float perThread_weightedRatePredict = perThread_ratePredict * weight_t;
            float perThread_weightedSpeedPredict = perThread_speedPredict * weight_t;
#if Debug_weightedPredictDistance
            pd_unweightedPredictListing[tIndex] = perThread_ratePredict;
#endif
            // write weighted predict. index: [rate/Speed][pred-h][search-h][win][k1]
            int isSpeed = 0;
            tIndex = (((isSpeed*predictStepLengthNr + predictStepLengthIndex)*searchStepLengthNr + searchStepLengthIndex)*windowSizeNr + windowSizeIndex)*kLevel1Nr + perThread_kLevel1Index;
            pd_allCombination_weightedPredictListing[tIndex] = perThread_weightedRatePredict;
            isSpeed = 1;
            tIndex = (((isSpeed*predictStepLengthNr + predictStepLengthIndex)*searchStepLengthNr + searchStepLengthIndex)*windowSizeNr + windowSizeIndex)*kLevel1Nr + perThread_kLevel1Index;
            pd_allCombination_weightedPredictListing[tIndex] = perThread_weightedSpeedPredict;
#elif !WeightExists_isTesting
            // see: gpu-r-exp_plan.docx ==>> section "For Analysis".
            // output index (query-aligned), for analysis: [predictStepLengthIndex][searchStepLengthIndex][windowSizeIndex][perThread_kLevel1Index]
            tIndex = ((predictStepLengthIndex*searchStepLengthNr + searchStepLengthIndex)*windowSizeNr + windowSizeIndex)*kLevel1Nr + perThread_kLevel1Index;

            // rank needs positive rmse values, here using rateError as prediction measurement (for sort in kernel 3).
#if Score_regardsE2d
            pd_perTimePoint_combinationMeasurementListing[tIndex] = sqrt(rateError * rateError + speedError * speedError);
#else
            pd_perTimePoint_combinationMeasurementListing[tIndex] = abs(rateError);
#endif
#endif
        }/* for each k1 end, localId_2D == kLevel1Nr */
        __syncthreads();
    }/* for each pred-h end, physical loop */
}

/* =============================================================== kernel 3: score =============================================================== */
/* =============================================================== kernel 3: score =============================================================== */
__global__ void init_pd_perTimePoint_combinationIndexListing(
    int *pd_perTimePoint_combinationIndexListing,
    int perTimePoint_combinationNr,
    int perPredictStepLength_combinationNr,
    int queryRecordOkForWeight){
    // fill values for sort_by_key's values
    if (globalId_2D < perTimePoint_combinationNr){
        if (queryRecordOkForWeight){
            pd_perTimePoint_combinationIndexListing[globalId_2D] = globalId_2D % perPredictStepLength_combinationNr;
        }
        else{
            pd_perTimePoint_combinationIndexListing[globalId_2D] = -999;
        }
    }
}

__global__ void updatePerTimePointRankToWeightListing(
    float *pd_perTimePoint_combinationMeasurementListing_perPredictSorted,
    int   *pd_perTimePoint_combinationIndexListing_perPredictSorted,
    float *pd_weightListing,
    int    flowIsGoingDown,
    int    flowIndicatorIndex,
    int    perPredictStepLength_combinationNr,
    int    perTimePoint_combinationNr)
{
    int perBlock_predictStepLengthIndex = blockId_col;

    // index in input
    int perThread_measurementIndex = globalId_3D;

    int perThread_indexToUpdateIn_perPredictStepLengthCombination = pd_perTimePoint_combinationIndexListing_perPredictSorted[perThread_measurementIndex];

    // index in output: [flowIsGoingDown][flowIndicatorIndex][pred-h][search-h = 0][win = 0][k1 = 0]
    int perBlock_currentPredictStepLength_weightIndexStart = (flowIsGoingDown*FlowIndicator_segment_nr + flowIndicatorIndex)*perTimePoint_combinationNr + perBlock_predictStepLengthIndex * perPredictStepLength_combinationNr;
    int perThread_weightIndexOffset = perThread_indexToUpdateIn_perPredictStepLengthCombination;
    int perThread_weightIndex = perBlock_currentPredictStepLength_weightIndexStart + perThread_weightIndexOffset;

    float perThread_valueToAdd;
#if Score_valueMethod != 0
    float perThread_distance = pd_perTimePoint_combinationMeasurementListing_perPredictSorted[localId_3D];
#endif
#if Score_valueMethod == 0   // default rank-linear
    int rank_t = localId_3D;
    perThread_valueToAdd = OneBlock_size3D * (1 - Score_RankingPunishNrRatio) - rank_t;
    if (perThread_valueToAdd < 0){
        perThread_valueToAdd *= Score_RankingPunishMultiply;
    }
#elif Score_valueMethod == 1 // dist linear reciprocal
    if (perThread_distance < Score_minDistanceForDivision) perThread_distance = Score_minDistanceForDivision;
    perThread_valueToAdd = 1.0 / perThread_distance;
#elif Score_valueMethod == 2 // dist pow_2  reciprocal
    if (perThread_distance < Score_minDistanceForDivision) perThread_distance = Score_minDistanceForDivision;
    perThread_valueToAdd = 1.0 / (perThread_distance * perThread_distance);
#elif Score_valueMethod == 3 // dist linear subtraction
    if (perThread_distance > Score_maxDistanceForSubtraction) perThread_distance = Score_maxDistanceForSubtraction;
    perThread_valueToAdd = Score_maxDistanceForSubtraction - perThread_distance;
#elif Score_valueMethod == 4 // dist pow_2  subtraction
    if (perThread_distance > Score_maxDistanceForSubtraction) perThread_distance = Score_maxDistanceForSubtraction;
    perThread_valueToAdd = Score_maxDistanceForSubtraction * Score_maxDistanceForSubtraction - perThread_distance * perThread_distance;
#elif Score_valueMethod == 5 // exponential of dist
    if (perThread_distance < Score_minDistanceForDivision) perThread_distance = Score_minDistanceForDivision;
    if (perThread_distance > Score_maxDistanceForSubtraction) perThread_distance = Score_maxDistanceForSubtraction;
    perThread_valueToAdd =  expf(-perThread_distance);
#elif Score_valueMethod == 6 // exponential of rank
    int rank_t = localId_3D;
    perThread_valueToAdd = expf(-rank_t);
#else
    static assert(false, "Unknown Distance Type");
#endif

    pd_weightListing[perThread_weightIndex] += perThread_valueToAdd;

}

/* =============================================================== kernel 4: normalize =============================================================== */
/* =============================================================== kernel 4: normalize =============================================================== */
__global__ void normalizeThresholdTreatment(float *pd_weightListing, float threshold)
{
    float weight_t = pd_weightListing[globalId];
    weight_t -= threshold;
    if (weight_t < 0.0f){
        weight_t = 0.0f;
    }
    pd_weightListing[globalId] = weight_t;
}

__global__ void normalizeDivision(float *pd_weightListingWithOffset, float sumTotal, float theSameTolerance)
{
    if (sumTotal < theSameTolerance){
        pd_weightListingWithOffset[localId] = 0.0f;
    }
    else{
        pd_weightListingWithOffset[localId] /= sumTotal;
    }
}

/* =============================================================== main =============================================================== */
/* =============================================================== main =============================================================== */
int main()
{
    printDateNTime();
    wbLog(TRACE, "Stage: ", Stage);

#if Write_ph_output
    char         p_char_outputBinFile[] = "../../gpu-r-proj/t145325_expResult.bin";
#endif
    char         p_char_inputMatFile[] = "t145325.clean_as.vector_single.mat";
    char         p_char_incidentFlagListingFile[] = "../../gpu-r-proj/incidentFlagBinary.bin";
    char         p_char_holidayFlagListingFile[] = "../../gpu-r-proj/holidayFlag.bin";
#if !WeightExists_isTesting && !NormalizeWeightOnly || NormalizeWeightOnly
    char         p_char_weightBeforeNormalizationFile[] = "../../gpu-r-proj/weightBeforeNormalizationFloat.bin";
#endif
    char         p_char_weightNormalizedFile[] = "../../gpu-r-proj/weightNormalizedFloat.bin";
#if Write_analysisFlag
    char         p_char_analysisFlagListingFile[] = "../../gpu-r-proj/analysisFlag.bin";
#endif
#if WeightExists_isTesting && !NormalizeWeightOnly
    char         p_char_weightedPredictFile[] = "../../gpu-r-proj/weightedPredict.bin";
#endif
    const char  *variableName;
    MATFile     *pMatFile;
    mxArray     *pMxArray;

    // host memory
    float *ph_input;
#if Write_ph_output
    float *ph_output;
#endif
#if Debug_weightedPredictDistance
    float *ph_unweightedPredictListing;
#endif
#if Debug_allDayDistance
    int    ph_searchStepLengthListing[] ={2};
    int    ph_windowSizeListing[]       ={0};
    int    ph_kLevel1Listing[]          ={3};
#elif !Debug_allDayDistance
    int    *ph_searchStepLengthListing;
    int    *ph_windowSizeListing;
    int    *ph_kLevel1Listing;
#endif
    int    ph_predictStepLengthListing[]= { 1, 2, 4, 8 };
    int    recordFlagListingByteSize, recordFlagNr;
    int    holidayFlagListingByteSize;
    int   *ph_incidentFlagScannedListing;
    int   *ph_holidayFlagListing;
    int   *ph_recordOkForAnalysisFlagListing;
    float *ph_weightListing;
#if WeightExists_isTesting
    float *ph_weightedPredictListing;
#elif !WeightExists_isTesting
#endif

    // device memory
    float *pd_input;
#if Write_ph_output
    float *pd_output;
#endif
#if Debug_weightedPredictDistance
    float *pd_unweightedPredictListing;
#endif
    float *pd_oneTimePoint_searchStepLength_searchShift_day_distance;
    int   *pd_oneTimePoint_searchStepLength_searchShift_day_index;
    float *pd_perTimePoint_combinationMeasurementListing;
    int   *pd_perTimePoint_combinationIndexListing;
    int   *pd_searchStepLengthListing;
    int   *pd_windowSizeListing;
    int   *pd_kLevel1Listing;
    int   *pd_predictStepLengthListing;
    int   *pd_incidentFlagScannedListing;
    float *pd_weightListing;
    float *pd_allCombination_weightedPredictListing;

    // debug var def
#if Debug_allDayDistance
    float *ph_oneTimePoint_searchStepLength_searchShift_day_distance;
    int   *ph_oneTimePoint_searchStepLength_searchShift_day_index;
#endif
#if Debug_sortedKnnPredict
    float *ph_sortedKnnPrediction;
    float *pd_sortedKnnPrediction;
    int sortedKnnPredictionNr, sortedKnnPredictionByteSize;
#endif
#if Debug_weightOfCombinationNIndex_txt
    float *ph_perTimePoint_combinationMeasurementListing;
    int   *ph_perTimePoint_combinationIndexListing;
#endif

    // algorithm param
#if Write_ph_output
    int resultRowNr;
    int outputNr;
    int outputByteSize;
#endif
    int inputRecordTotalNr;
    int completeDayNr, nonCompleteDayNr, searchableDayNr;
    int inputByteSize;

    int    searchStepLengthNr = GetArrayLength(ph_searchStepLengthListing);
    int    windowSizeNr = GetArrayLength(ph_windowSizeListing);
    int    kLevel1Nr = GetArrayLength(ph_kLevel1Listing);
    int    predictStepLengthNr = GetArrayLength(ph_predictStepLengthListing);

    // algorithm & analysis param
    int perTimePoint_combinationNr; // rate & speed are NOT considered as combination factor, though they are part of the array/index.

    // analysis param
    int perTimePoint_combinationMeasurementByteSize;
    int perPredictStepLength_combinationNr;
    int weightNr;

    // read input: mat
    pMatFile = matOpen(p_char_inputMatFile, "r"); //r: read only; w6: write v5
    if (pMatFile == NULL) {printf("Error opening %s\n", p_char_inputMatFile);return(1);}
    pMxArray = matGetNextVariable(pMatFile, &variableName);
    if (pMxArray == NULL) {printf("Error matGetNextVariable, 'pMxArray == NULL', p_char_inputMatFile: %s\n", p_char_inputMatFile);return(1);}
    cout<<"name: " << variableName <<
        ", dim: " << mxGetNumberOfDimensions(pMxArray) <<
        ", size: " << mxGetElementSize(pMxArray) <<
        ", elementNr: " << mxGetNumberOfElements(pMxArray) <<   // both real and imag (if any) values are count.
        ", rowNr: " << mxGetM(pMxArray) <<                      // vector is stored as row-incremental. 
        ", colNr: " << mxGetN(pMxArray) <<
        "\n";
    ph_input = (float *)mxGetData(pMxArray);
    
    pMxArray = matGetNextVariable(pMatFile, &variableName);
    if (pMxArray == NULL) {printf("Error matGetNextVariable ph_searchStepLengthListing, 'pMxArray == NULL', p_char_inputMatFile: %s\n", p_char_inputMatFile);return(1);}
    ph_searchStepLengthListing = (int *)mxGetData(pMxArray);
    
    pMxArray = matGetNextVariable(pMatFile, &variableName);
    if (pMxArray == NULL) {printf("Error matGetNextVariable ph_windowSizeListing, 'pMxArray == NULL', p_char_inputMatFile: %s\n", p_char_inputMatFile);return(1);}
    ph_windowSizeListing = (int *)mxGetData(pMxArray);
    
    pMxArray = matGetNextVariable(pMatFile, &variableName);
    if (pMxArray == NULL) {printf("Error matGetNextVariable ph_kLevel1Listing, 'pMxArray == NULL', p_char_inputMatFile: %s\n", p_char_inputMatFile);return(1);}
    ph_kLevel1Listing = (int *)mxGetData(pMxArray);
    
#if Debug_show_param
    for (int i = 0; i < searchStepLengthNr; i++){
        wbLog(TRACE, "ph_searchStepLengthListing: ", ph_searchStepLengthListing[i]);
    }
    for (int i = 0; i < windowSizeNr; i++){
        wbLog(TRACE, "ph_windowSizeListing: ", ph_windowSizeListing[i]);
    }
    for (int i = 0; i < kLevel1Nr; i++){
        wbLog(TRACE, "ph_kLevel1Listing: ", ph_kLevel1Listing[i]);
    }
    for (int i = 0; i < predictStepLengthNr; i++){
        wbLog(TRACE, "ph_predictStepLengthListing: ", ph_predictStepLengthListing[i]);
    }
#endif

    // init param
    perTimePoint_combinationNr = searchStepLengthNr * windowSizeNr * kLevel1Nr * predictStepLengthNr;
    perTimePoint_combinationMeasurementByteSize = perTimePoint_combinationNr * sizeof(float);
    wbLog(TRACE, "perTimePoint_combinationNr: ", perTimePoint_combinationNr);
    perPredictStepLength_combinationNr = searchStepLengthNr * windowSizeNr * kLevel1Nr;
    inputRecordTotalNr = (int)mxGetNumberOfElements(pMxArray) / 2;
    completeDayNr = inputRecordTotalNr / Daily_record_nr;
    nonCompleteDayNr = (inputRecordTotalNr - 1) / Daily_record_nr + 1; // ceil
    wbLog(TRACE, "inputRecordTotalNr: ", inputRecordTotalNr, ", completeDayNr: ", completeDayNr, ", nonCompleteDayNr: ", nonCompleteDayNr);
    searchableDayNr = completeDayNr - 2;
    inputByteSize = inputRecordTotalNr * 2 * sizeof(float);
#if Write_ph_output
    resultRowNr = completeDayNr * Daily_record_nr; // not fully used.
    outputNr = resultRowNr * perTimePoint_combinationNr * Data_variable_nr;
    outputByteSize = outputNr * sizeof(float);
#endif
    weightNr = DailyUpDown_segment_nr * FlowIndicator_segment_nr * predictStepLengthNr * perPredictStepLength_combinationNr;
    if (Float_max / (float)(perPredictStepLength_combinationNr^2) < searchableDayNr * 288/2/2){ // (a). using formula n^2/2; (b). up&down, so divided again by 2.
        wbLog(WARN, "Sum of weights after kernel_3 can be beyond Float_max in the extreme situation that all flow-indicators are the same.");
    }

    // read incident flag
    recordFlagListingByteSize = getFileByteSizeByLocation(p_char_incidentFlagListingFile);
    if (((float)recordFlagListingByteSize)/(float)sizeof(int) != (float)nonCompleteDayNr * Daily_record_nr){
        wbLog(ERROR, "wrong p_char_incidentFlagListingFile size, should contain: ", nonCompleteDayNr * Daily_record_nr, " int values.");
    }
    recordFlagNr = recordFlagListingByteSize/sizeof(int);
    ifstream incidentFlagListingStream(p_char_incidentFlagListingFile, ifstream::binary);
    ph_incidentFlagScannedListing = new int[recordFlagNr];
    incidentFlagListingStream.read((char*)ph_incidentFlagScannedListing, recordFlagListingByteSize);
    wbTime_start(GPU, "Scan incident flag.");
    thrust::inclusive_scan(ph_incidentFlagScannedListing, ph_incidentFlagScannedListing + recordFlagNr, ph_incidentFlagScannedListing); // GetArrayLength doesn't work
    wbTime_stop(GPU, "Scan incident flag.");

    // read holiday flag
    holidayFlagListingByteSize = getFileByteSizeByLocation(p_char_holidayFlagListingFile);
    if ((float)holidayFlagListingByteSize/sizeof(int) != (float)nonCompleteDayNr){
        wbLog(ERROR, "wrong p_char_holidayFlagListingFile size, should contain: ", nonCompleteDayNr, " int values.");
    }
    ifstream holidayFlagListingStream(p_char_holidayFlagListingFile, ifstream::binary);
    ph_holidayFlagListing = new int[holidayFlagListingByteSize/sizeof(int)];
    holidayFlagListingStream.read((char*)ph_holidayFlagListing, holidayFlagListingByteSize);

    // init analysis flag
    ph_recordOkForAnalysisFlagListing = new int[recordFlagNr];
    memset(ph_recordOkForAnalysisFlagListing, 0, recordFlagListingByteSize); // init with 0

    //@@ Allocate GPU memory
    wbTime_start(GPU, "Allocating GPU memory.");
    Check(cudaMalloc((void **)&pd_input, inputByteSize));
#if Write_ph_output
    Check(cudaMalloc((void **)&pd_output, outputByteSize));
#endif
#if Debug_weightedPredictDistance
    Check(cudaMalloc((void **)&pd_unweightedPredictListing, weightNr * sizeof(float)));
#endif
    Check(cudaMalloc((void **)&pd_perTimePoint_combinationMeasurementListing, perTimePoint_combinationMeasurementByteSize));
    Check(cudaMalloc((void **)&pd_perTimePoint_combinationIndexListing, perTimePoint_combinationNr * sizeof(int)));
    int oneTimePointAllNeighborDistanceNr = searchStepLengthNr * (ph_windowSizeListing[windowSizeNr-1] * 2 + 1) * searchableDayNr;
    Check(cudaMalloc((void **)&pd_oneTimePoint_searchStepLength_searchShift_day_distance, oneTimePointAllNeighborDistanceNr * sizeof(float))); // ~ 900KB for max config combinations
    Check(cudaMalloc((void **)&pd_oneTimePoint_searchStepLength_searchShift_day_index, oneTimePointAllNeighborDistanceNr * sizeof(int)));
    Check(cudaMalloc((void **)&pd_searchStepLengthListing, searchStepLengthNr * sizeof(int)));
    Check(cudaMalloc((void **)&pd_windowSizeListing, windowSizeNr * sizeof(int)));
    Check(cudaMalloc((void **)&pd_kLevel1Listing, kLevel1Nr * sizeof(int)));
    Check(cudaMalloc((void **)&pd_predictStepLengthListing, predictStepLengthNr * sizeof(int)));
    Check(cudaMalloc((void **)&pd_incidentFlagScannedListing, nonCompleteDayNr * Daily_record_nr * sizeof(int)));
    Check(cudaMalloc((void **)&pd_weightListing, weightNr * sizeof(float)));
    Check(cudaMalloc((void **)&pd_allCombination_weightedPredictListing, weightNr / (DailyUpDown_segment_nr * FlowIndicator_segment_nr) * Data_variable_nr * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");
    wbLog(TRACE, "oneTimePointAllNeighborDistanceNr: ", oneTimePointAllNeighborDistanceNr);
    wbLog(TRACE, "Default_distance_of_day: ", Default_distance_of_day);
    //@@ Allocate GPU memory - end

    //@@ Allocate Host memory
    wbTime_start(Generic, "Allocating Host memory.");
#if Write_ph_output
    ph_output = (float *)malloc(outputByteSize);
#endif
    ph_weightListing = new float[weightNr];
#if Debug_weightedPredictDistance
    ph_unweightedPredictListing = new float[weightNr];
#endif
#if WeightExists_isTesting
    // read weight
    char *p_char_weightInputFile_t;
#if NormalizeWeightOnly
    p_char_weightInputFile_t = p_char_weightBeforeNormalizationFile;
#elif !NormalizeWeightOnly
    p_char_weightInputFile_t = p_char_weightNormalizedFile;
#endif
    int weightListingByteSize = getFileByteSizeByLocation(p_char_weightInputFile_t);
    if (weightListingByteSize != weightNr * sizeof(float)) wbLog(ERROR, "weightListing file size does not match knn-param config");
    ifstream weightListingStream(p_char_weightInputFile_t, ifstream::binary);
    weightListingStream.read((char*)ph_weightListing, weightListingByteSize);
    weightListingStream.close();
#if Debug_weightedPredictDistance
    wbLog(TRACE, "read weight");
    for (int i = 0; i < weightNr; i++){
        cout << ph_weightListing[i] << "\t";
        if (i > 0 && (i + 1) % perPredictStepLength_combinationNr == 0)
            cout << "\n";
    }
    cout << "\n";
#endif
    // to save predict
    ph_weightedPredictListing = new float[completeDayNr * Daily_record_nr * Data_variable_nr * predictStepLengthNr];
    memset_float_sunny(ph_weightedPredictListing, -999.0f, completeDayNr * Daily_record_nr * Data_variable_nr * predictStepLengthNr);
#elif !WeightExists_isTesting
#endif
    wbTime_stop(Generic, "Allocating Host memory.");
    //@@ Allocate Host memory - End

    //@@ Allocate debug memory
#if Debug_allDayDistance
    ph_oneTimePoint_searchStepLength_searchShift_day_distance = (float *)malloc(oneTimePointAllNeighborDistanceNr * sizeof(float));
    ph_oneTimePoint_searchStepLength_searchShift_day_index    = (int *)  malloc(oneTimePointAllNeighborDistanceNr * sizeof(int));
    std::remove("../../gpu-r-proj/Debug_allDayDistance_append_kernel1.bin");
    if (std::ifstream("../../gpu-r-proj/Debug_allDayDistance_append_kernel1.bin")) wbLog(ERROR, "failed to delete file: Debug_allDayDistance_append_kernel1.bin");
#endif
#if Debug_sortedKnnPredict
    sortedKnnPredictionNr = windowSizeNr * searchStepLengthNr * 256 * 2;
    sortedKnnPredictionByteSize = sortedKnnPredictionNr * sizeof(float);
    ph_sortedKnnPrediction = (float *)malloc(sortedKnnPredictionByteSize);
    Check(cudaMalloc((void **)&pd_sortedKnnPrediction, sortedKnnPredictionByteSize));
    std::remove("../../gpu-r-proj/Debug_sortedKnnPredict_append_kernel2.bin");
    if (std::ifstream("../../gpu-r-proj/Debug_sortedKnnPredict_append_kernel2.bin")) wbLog(ERROR, "failed to delete file: Debug_sortedKnnPredict_append_kernel2.bin");
#endif
#if Debug_weightOfCombinationNIndex_txt
    ph_perTimePoint_combinationMeasurementListing = (float *)malloc(perTimePoint_combinationMeasurementByteSize);
    ph_perTimePoint_combinationIndexListing = (int *)malloc(perTimePoint_combinationNr * sizeof(int));
#endif
    //@@ Allocate debug memory - End

    //@@ Copy memory to GPU
    wbTime_start(GPU, "Copying to GPU.");
    Check(cudaMemcpy(pd_input, ph_input,
        inputByteSize, cudaMemcpyHostToDevice));
    Check(cudaMemcpy(pd_searchStepLengthListing, ph_searchStepLengthListing,
        searchStepLengthNr * sizeof(int), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(pd_windowSizeListing, ph_windowSizeListing,
        windowSizeNr * sizeof(int), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(pd_kLevel1Listing, ph_kLevel1Listing,
        kLevel1Nr * sizeof(int), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(pd_predictStepLengthListing, ph_predictStepLengthListing,
        predictStepLengthNr * sizeof(int), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(pd_incidentFlagScannedListing, ph_incidentFlagScannedListing,
        nonCompleteDayNr * Daily_record_nr * sizeof(int), cudaMemcpyHostToDevice));
#if Write_ph_output
    cudaMemsetFloat_sunny(pd_output, -999.0f, outputNr);
#endif
#if WeightExists_isTesting
    Check(cudaMemcpy(pd_weightListing, ph_weightListing,
        weightNr * sizeof(float), cudaMemcpyHostToDevice));
#elif !WeightExists_isTesting
    cudaMemsetFloat_sunny(pd_weightListing, 0.0f, weightNr);
#endif
    cudaMemsetFloat_sunny(pd_perTimePoint_combinationMeasurementListing, 999.0f, perTimePoint_combinationNr);
    cudaMemsetFloat_sunny(pd_oneTimePoint_searchStepLength_searchShift_day_distance, Default_distance_of_day, oneTimePointAllNeighborDistanceNr);
    wbTime_stop(GPU, "Copying to GPU.");
    //@@ Copy memory to GPU - end

    //@@ Initialize the grid and block dimensions
    dim3 DimGrid(searchableDayNr, 1, 1);
    dim3 DimBlock(Block_size, 1, 1);

    dim3 DimGrid_2_pre((oneTimePointAllNeighborDistanceNr - 1)/256 + 1, 1, 1);
    dim3 DimBlock_2_pre(16, 16, 1);

    dim3 DimGrid_2(1, 1, 1);    // row & col: only 1 row & 1 col. searchStepLengthIndex & windowSizeIndex is in host physical loop
    dim3 DimBlock_2(16, 16, 1); // row: kLevel1Nr; col: predictStepLengthNr  // max: make sure can sort pair in a block
    if ((max(predictStepLengthNr, windowSizeNr) > 16) || max(kLevel1Nr, searchStepLengthNr) > 16) wbLog(ERROR, "Unexpcted large config.");

    dim3 DimGrid_3_pre((perTimePoint_combinationNr - 1)/256 + 1, 1, 1);
    dim3 DimBlock_3_pre(16, 16, 1);
    
    dim3 DimGrid_3(predictStepLengthNr, 1, 1);
    dim3 DimBlock_3(searchStepLengthNr, windowSizeNr, kLevel1Nr);

    dim3 DimGrid_4_threshold((weightNr - 1)/256 + 1, 1, 1);
    dim3 DimBlock_4_threshold(16, 16, 1);
    
    dim3 DimGrid_4_division(1, 1, 1);
    dim3 DimBlock_4_division(perPredictStepLength_combinationNr, 1, 1);
    //@@ Initialize the grid and block dimensions - end

    //@@ Launch GPU Kernels
    wbTime_start(GPU, "Main GPU computation");
    Check(cudaDeviceSynchronize());
    /* for day of history */
    int trainingDayNr = (int)(searchableDayNr * TrainingDataRatio);
    int queryDayIndex_inAllGlobalDays_init;
    int queryDayIndex_inAllGlobalDays_stop;
#if WeightExists_isTesting
    queryDayIndex_inAllGlobalDays_init = 1 + trainingDayNr;
    queryDayIndex_inAllGlobalDays_stop = searchableDayNr;
#elif !WeightExists_isTesting
    queryDayIndex_inAllGlobalDays_init = 1;
    queryDayIndex_inAllGlobalDays_stop = trainingDayNr;
#endif
    for (int queryDayIndex_inAllGlobalDays = queryDayIndex_inAllGlobalDays_init; queryDayIndex_inAllGlobalDays <= queryDayIndex_inAllGlobalDays_stop; queryDayIndex_inAllGlobalDays++){ // should be 1.inc ~ searchableDayNr.inc
#if Debug_queryDayIndex && !Debug_recordOffset_ofDay
        wbLog(TRACE, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays);
#endif
#if Debug_detail_timer
        if (queryDayIndex_inAllGlobalDays % 10 == 1){
            wbTime_start(Generic, "One part: 10-day computation");
        }
#endif

        // holiday-flag to weight & analysis flags
        int queryDayHolidayFlag = 0;
        int queryDayIsHoliday   = -1;
        int dayOrderInWorkDays  = -1; // "order" starts from 1. not "index" (which will start from 0)
        queryDayHolidayFlag = ph_holidayFlagListing[queryDayIndex_inAllGlobalDays];
        queryDayIsHoliday = (queryDayHolidayFlag > 0) ? 1 : 0;
#if Debug_analysisFlagListing_queryData
        if (queryDayIsHoliday) wbLog(DEBUG, "today is holiday");
#endif
        if (queryDayIsHoliday){
            dayOrderInWorkDays = 0;
        }
        else
        {
            dayOrderInWorkDays = getOnePlaceDigit(queryDayHolidayFlag);
#if Debug_analysisFlagListing_queryData
            if (dayOrderInWorkDays == 1) wbLog(DEBUG, "today is the 1st work day");
#endif
        }

        /* for record of day */
        int recordOffset_ofDay_init = 0;
        for (int recordOffset_ofDay = recordOffset_ofDay_init; recordOffset_ofDay < 288; recordOffset_ofDay++){ // should be 0.inc ~ 288.no.inc
#if Debug_recordOffset_ofDay
            wbLog(TRACE, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". recordOffset_ofDay: ", recordOffset_ofDay);
#endif

            // incident-flag to weight & analysis flags
            int recordOffsetIndexGlobal = queryDayIndex_inAllGlobalDays * Daily_record_nr + recordOffset_ofDay;
            int incidentNrInSelectedRange   = -1;
            int tooManyHistoryIncident      = 0;
            int tooManyPredictIncident      = 0;
            // query data incident situation
            for (int index = 0; index < searchStepLengthNr; index++){
                incidentNrInSelectedRange = ph_incidentFlagScannedListing[recordOffsetIndexGlobal - 1] - ph_incidentFlagScannedListing[recordOffsetIndexGlobal - 1 - ph_searchStepLengthListing[index]];
                if ((float)incidentNrInSelectedRange > IncidentToleranceRatio * (float)ph_searchStepLengthListing[index]){ // no need to consider win, cuz the query data is not influenced by win
                    tooManyHistoryIncident = 1;
#if Debug_analysisFlagListing_queryData
                    wbLog(DEBUG, "history incident_nr too high: ", incidentNrInSelectedRange, ". search-h: ", ph_searchStepLengthListing[index], ". ceiling incident nr: ", ph_incidentFlagScannedListing[recordOffsetIndexGlobal - 1], ". bottom nr: ", ph_incidentFlagScannedListing[recordOffsetIndexGlobal - 1 - ph_searchStepLengthListing[index]]);
#endif
                }
            }
            // ground truth incident situation
            incidentNrInSelectedRange = ph_incidentFlagScannedListing[recordOffsetIndexGlobal - 1 + ph_predictStepLengthListing[predictStepLengthNr - 1]] - ph_incidentFlagScannedListing[recordOffsetIndexGlobal - 1];
            if (incidentNrInSelectedRange > 0){
                tooManyPredictIncident = 1;
#if Debug_analysisFlagListing_queryData
                wbLog(DEBUG, "ground truth has incident");
#endif
            }

            // due to the new offset index which is modified to the same as ph_output, first severl records are not suitable for analysis
            int maxPredictLength_for_analysisFlag_t = ph_predictStepLengthListing[predictStepLengthNr - 1];
            int inInitMaxPredictLength = -1;
            if (queryDayIndex_inAllGlobalDays == queryDayIndex_inAllGlobalDays_init && recordOffset_ofDay - recordOffset_ofDay_init < maxPredictLength_for_analysisFlag_t - 1) inInitMaxPredictLength = 1;
            else inInitMaxPredictLength = 0;

            int queryRecordOkForAnalysis = -1;
            if (inInitMaxPredictLength || tooManyHistoryIncident || tooManyPredictIncident || queryDayIsHoliday || dayOrderInWorkDays == 1) queryRecordOkForAnalysis = 0;
            else queryRecordOkForAnalysis = 1;
            int queryRecordOkForWeight = -1;
            if (tooManyHistoryIncident || tooManyPredictIncident || queryDayIsHoliday || dayOrderInWorkDays == 1) queryRecordOkForWeight = 0;
            else queryRecordOkForWeight = 1;

            if (queryRecordOkForAnalysis){
                ph_recordOkForAnalysisFlagListing[recordOffsetIndexGlobal] = 1; // init-ed with 0
            }
#if Debug_analysisFlagListing_queryData
            else wbLog(DEBUG, "NOT ok for analysis");
#endif
            if (inInitMaxPredictLength < 0 || queryDayHolidayFlag == 0 || queryDayIsHoliday < 0 || dayOrderInWorkDays < 0 || incidentNrInSelectedRange < 0 || queryRecordOkForWeight < 0 || queryRecordOkForAnalysis < 0){
                wbLog(ERROR, "all situations should be covered, should not be the init values");
                wbLog(ERROR, "inInitMaxPredictLength:    ", inInitMaxPredictLength);
                wbLog(ERROR, "queryDayHolidayFlag:       ", queryDayHolidayFlag);
                wbLog(ERROR, "queryDayIsHoliday:         ", queryDayIsHoliday);
                wbLog(ERROR, "dayOrderInWorkDays:        ", dayOrderInWorkDays);
                wbLog(ERROR, "incidentNrInSelectedRange: ", incidentNrInSelectedRange);
                wbLog(ERROR, "queryRecordOkForWeight:    ", queryRecordOkForWeight);
                wbLog(ERROR, "queryRecordOkForAnalysis:  ", queryRecordOkForAnalysis);
            }

#if Do_kernel
            // kernel 1
            calculateOneTimePoint_searchShiftNSearchStepLength <<< DimGrid, DimBlock >>> (
                queryDayIndex_inAllGlobalDays, recordOffset_ofDay,
#if Write_ph_output
                pd_output,
#endif
                pd_oneTimePoint_searchStepLength_searchShift_day_distance,
                pd_input, inputRecordTotalNr, completeDayNr, searchableDayNr,
                pd_incidentFlagScannedListing,
                pd_searchStepLengthListing, searchStepLengthNr,
                pd_windowSizeListing, windowSizeNr,
                pd_kLevel1Listing, kLevel1Nr,
                pd_predictStepLengthListing, predictStepLengthNr);
            Check(cudaDeviceSynchronize());
#if Debug_allDayDistance
            Check(cudaMemcpy(ph_oneTimePoint_searchStepLength_searchShift_day_distance, pd_oneTimePoint_searchStepLength_searchShift_day_distance, oneTimePointAllNeighborDistanceNr * sizeof(float), cudaMemcpyDeviceToHost)); // ~ 900KB
            if (recordOffset_ofDay % 100 == 0){
                ofstream stream_outFile_debug_allDayDistance_append_kernel1("../../gpu-r-proj/Debug_allDayDistance_append_kernel1.bin", ios::out | ios::binary | ios::app);
                if (stream_outFile_debug_allDayDistance_append_kernel1.good()){
                    stream_outFile_debug_allDayDistance_append_kernel1.write((char *)ph_oneTimePoint_searchStepLength_searchShift_day_distance, oneTimePointAllNeighborDistanceNr * sizeof(float));
                }
                else{
                    wbLog(ERROR, "!stream_outFile_debug_allDayDistance_append_kernel1.good: Debug_allDayDistance_append_kernel1.bin");
                }
                stream_outFile_debug_allDayDistance_append_kernel1.close();
            }
#endif

            // for kernel 2 & 3 segment index
            int flowIsGoingDown, flowIndicatorIndex;
            if(Score_useFlowSegment){
                if (recordOffset_ofDay >= 12*3 && recordOffset_ofDay < 12*15){ // reason: hourlyCenters.png
                    flowIsGoingDown = 0;
                }
                else{
                    flowIsGoingDown = 1;
                }                
            }
            else{
                flowIsGoingDown = 0;
            }
            int recordOffset_flowIndicator_index_t = recordOffsetIndexGlobal * Data_variable_nr;
            float tFlow = (ph_input[recordOffset_flowIndicator_index_t - 2] + ph_input[recordOffset_flowIndicator_index_t - 4] + ph_input[recordOffset_flowIndicator_index_t - 6]) / 3.0f;
            if(Score_useFlowSegment){
                flowIndicatorIndex = (int)tFlow / 10; //TODO: improve: (a) frequency segment  (b) improve: not just assume 0~100
                flowIndicatorIndex = min(flowIndicatorIndex, FlowIndicator_segment_nr - 1);
            }
            else{
                flowIndicatorIndex = 0;
            }
#if Debug_queryRecord_flowIndicator
            wbLog(TRACE, "tFlow: ", tFlow, ". flowIsGoingDown: ", flowIsGoingDown, ". flowIndicatorIndex: ", flowIndicatorIndex);
#endif

            // kernel 2
            init_pd_oneTimePoint_searchStepLength_searchShift_day_index <<< DimGrid_2_pre, DimBlock_2_pre >>> (pd_oneTimePoint_searchStepLength_searchShift_day_index, oneTimePointAllNeighborDistanceNr);
            Check(cudaDeviceSynchronize());
            for (int searchStepLengthIndex= 0; searchStepLengthIndex < searchStepLengthNr; searchStepLengthIndex++){
                for (int windowSizeIndex = 0; windowSizeIndex < windowSizeNr; windowSizeIndex++){
                    int windowSize = ph_windowSizeListing[windowSizeIndex];
                    int windowSize_max = ph_windowSizeListing[windowSizeNr - 1];
                    int searchShiftNr_max = windowSize_max * 2 + 1;
                    int searchShiftIndex_inMaxWindow_left = -windowSize + windowSize_max;
                    int searchShiftIndex_inMaxWindow_right = windowSize + windowSize_max;
                    int dataToCopy_oneTimePointDistance_index_left  = (searchStepLengthIndex * searchShiftNr_max + searchShiftIndex_inMaxWindow_left)  * searchableDayNr + 0; // [searchStepLengthIndex][left] [dayIndex_in_analysedDays == 0]
                    int dataToCopy_oneTimePointDistance_index_right = (searchStepLengthIndex * searchShiftNr_max + searchShiftIndex_inMaxWindow_right) * searchableDayNr + (searchableDayNr - 1); // [searchStepLengthIndex][right][dayIndex_in_analysedDays == last]
                    
                    // sort
#if Debug_sortedKnnPredict
                    wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". ", "recordOffset_ofDay: ", recordOffset_ofDay, ". ", "kernel 2 debug before dist-sort");
                    Check(cudaMemcpy(ph_oneTimePoint_searchStepLength_searchShift_day_distance, pd_oneTimePoint_searchStepLength_searchShift_day_distance, oneTimePointAllNeighborDistanceNr * sizeof(float), cudaMemcpyDeviceToHost)); // ~ 900KB
                    Check(cudaMemcpy(ph_oneTimePoint_searchStepLength_searchShift_day_index, pd_oneTimePoint_searchStepLength_searchShift_day_index, oneTimePointAllNeighborDistanceNr * sizeof(int), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < oneTimePointAllNeighborDistanceNr; i++){
                        cout << ph_oneTimePoint_searchStepLength_searchShift_day_index[i] << "\t" << ph_oneTimePoint_searchStepLength_searchShift_day_distance[i] << "\n";
                    }
#endif
                    thrust::device_ptr<float> p_key_ofNeighborDistance(pd_oneTimePoint_searchStepLength_searchShift_day_distance);
                    thrust::device_ptr<int>   p_data_ofNeighborDistance(pd_oneTimePoint_searchStepLength_searchShift_day_index);
                    thrust::sort_by_key(p_key_ofNeighborDistance + dataToCopy_oneTimePointDistance_index_left,
                        p_key_ofNeighborDistance + dataToCopy_oneTimePointDistance_index_left + (dataToCopy_oneTimePointDistance_index_right - dataToCopy_oneTimePointDistance_index_left + 1),
                        p_data_ofNeighborDistance + dataToCopy_oneTimePointDistance_index_left);
                    Check(cudaDeviceSynchronize());
#if Debug_sortedKnnPredict
                    wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". ", "recordOffset_ofDay: ", recordOffset_ofDay, ". ", "kernel 2 debug after dist-sort");
                    Check(cudaMemcpy(ph_oneTimePoint_searchStepLength_searchShift_day_distance, pd_oneTimePoint_searchStepLength_searchShift_day_distance, oneTimePointAllNeighborDistanceNr * sizeof(float), cudaMemcpyDeviceToHost)); // ~ 900KB
                    Check(cudaMemcpy(ph_oneTimePoint_searchStepLength_searchShift_day_index, pd_oneTimePoint_searchStepLength_searchShift_day_index, oneTimePointAllNeighborDistanceNr * sizeof(int), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < oneTimePointAllNeighborDistanceNr; i++){
                        cout << ph_oneTimePoint_searchStepLength_searchShift_day_index[i] << "\t" << ph_oneTimePoint_searchStepLength_searchShift_day_distance[i] << "\n";
                    }
#endif

                    // call kernel 2
                    calculateOneTimePoint_kLevel1NPredictStepLength <<< DimGrid_2, DimBlock_2 >>> (
#if Debug_sortedKnnPredict
                        pd_sortedKnnPrediction,
#endif
#if Debug_weightedPredictDistance
                        pd_unweightedPredictListing,
#endif
#if WeightExists_isTesting
                        pd_weightListing, flowIsGoingDown, flowIndicatorIndex, pd_allCombination_weightedPredictListing,
#elif !WeightExists_isTesting
                        queryRecordOkForWeight,
#endif
                        searchStepLengthIndex, windowSizeIndex,
                        pd_perTimePoint_combinationMeasurementListing,
                        queryDayIndex_inAllGlobalDays, recordOffset_ofDay,
#if Write_ph_output
                        pd_output,
#endif
                        pd_oneTimePoint_searchStepLength_searchShift_day_distance,
                        pd_oneTimePoint_searchStepLength_searchShift_day_index,
                        pd_input, inputRecordTotalNr, completeDayNr, searchableDayNr,
                        pd_searchStepLengthListing, searchStepLengthNr,
                        pd_windowSizeListing, windowSizeNr,
                        pd_kLevel1Listing, kLevel1Nr,
                        pd_predictStepLengthListing, predictStepLengthNr);
                    Check(cudaDeviceSynchronize());
#if Debug_sortedKnnPredict
                    if (recordOffset_ofDay % 100 == 0){
                        Check(cudaMemcpy(ph_sortedKnnPrediction, pd_sortedKnnPrediction, sortedKnnPredictionByteSize, cudaMemcpyDeviceToHost));
                        ofstream stream_outFile_debug_sortedKnnPredict_append_kernel2_file("../../gpu-r-proj/Debug_sortedKnnPredict_append_kernel2.bin", ios::out | ios::binary | ios::app);
                        if (stream_outFile_debug_sortedKnnPredict_append_kernel2_file.good()){
                            stream_outFile_debug_sortedKnnPredict_append_kernel2_file.write((char *)ph_sortedKnnPrediction, sortedKnnPredictionByteSize);
                        }
                        else{
                            wbLog(ERROR, "!stream_outFile_debug_sortedKnnPredict_append_kernel2_file.good: Debug_sortedKnnPredict_append_kernel2.bin");
                        }
                        stream_outFile_debug_sortedKnnPredict_append_kernel2_file.close();
                    }
#endif
                } /* for each win */
            } /* for each search-h */
#if Debug_weightedPredictDistance
            if (recordOffset_ofDay % 100 == 0){
                wbLog(TRACE, "recordOffset_ofDay: ", recordOffset_ofDay, ". Unweighted Predict");
                Check(cudaMemcpy(ph_unweightedPredictListing, pd_unweightedPredictListing, weightNr * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < weightNr; i++){
                    cout << ph_unweightedPredictListing[i] << "\t";
                    if (i > 0 && (i + 1) % perPredictStepLength_combinationNr == 0)
                        cout << "\n";
                }
                cout << "\n";
            }
#endif

#if WeightExists_isTesting
            thrust::device_ptr<float> p_allCombination_weightedPredictListing(pd_allCombination_weightedPredictListing);
            for (int isSpeed = 0; isSpeed <= 1; isSpeed++){
                for (int predict_index_t = 0; predict_index_t < predictStepLengthNr; predict_index_t++){
                    // reduce to get weighted predict, to host. index: [rate/Speed][pred-h][search-h][win][k1]
                    int offset_t = (isSpeed*predictStepLengthNr + predict_index_t) * searchStepLengthNr * windowSizeNr * kLevel1Nr;
                    float sumTotal_as_weightedPredict = thrust::reduce(p_allCombination_weightedPredictListing + offset_t, p_allCombination_weightedPredictListing + offset_t + searchStepLengthNr * windowSizeNr * kLevel1Nr);
                    // write index: [recordOffsetIndexGlobal + (pred-h - 1) ][pred-h][rate/Speed]
                    int weightedPredictIndex = ((recordOffsetIndexGlobal + ph_predictStepLengthListing[predict_index_t] - 1)*predictStepLengthNr + predict_index_t)*Data_variable_nr + isSpeed;
                    ph_weightedPredictListing[weightedPredictIndex] = sumTotal_as_weightedPredict;
                    // no measurement needed, will do in R
                }
            }
            //TODO: optional: measure deviation
#elif !WeightExists_isTesting
            // kernel 3: sort key(measurement)-value(index) & update weight
            // kernel 3 pre: init
            init_pd_perTimePoint_combinationIndexListing <<< DimGrid_3_pre, DimBlock_3_pre >>> (
                pd_perTimePoint_combinationIndexListing,
                perTimePoint_combinationNr,
                perPredictStepLength_combinationNr,
                queryRecordOkForWeight);
#if Debug_weightOfCombinationNIndex_txt
            if (recordOffset_ofDay % 100 == 0){
                wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". ", "recordOffset_ofDay: ", recordOffset_ofDay, ". ", "kernel 3 debug before sort");
                Check(cudaMemcpy(ph_perTimePoint_combinationMeasurementListing, pd_perTimePoint_combinationMeasurementListing, perTimePoint_combinationMeasurementByteSize, cudaMemcpyDeviceToHost));
                Check(cudaMemcpy(ph_perTimePoint_combinationIndexListing, pd_perTimePoint_combinationIndexListing, perTimePoint_combinationNr * sizeof(int), cudaMemcpyDeviceToHost));
                for (int i = 0; i < perTimePoint_combinationNr; i++){
                    cout << ph_perTimePoint_combinationIndexListing[i] << "\t" << ph_perTimePoint_combinationMeasurementListing[i] << "\n";
                }
            }
#endif
            if (queryRecordOkForWeight){
                int offset_kernel3_t;
                thrust::device_ptr<float> p_key_ofCombination(pd_perTimePoint_combinationMeasurementListing);
                thrust::device_ptr<int>   p_data_ofCombination(pd_perTimePoint_combinationIndexListing);
                for (int index = 0; index < predictStepLengthNr; index++){
                    offset_kernel3_t = index * perPredictStepLength_combinationNr;
                    thrust::sort_by_key(p_key_ofCombination + offset_kernel3_t, p_key_ofCombination + offset_kernel3_t + perPredictStepLength_combinationNr, p_data_ofCombination + offset_kernel3_t);
                }
            }
#if Debug_weightOfCombinationNIndex_txt
            else
                wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". ", "recordOffset_ofDay: ", recordOffset_ofDay, ". ", "kernel 3 debug skipping, not ok for weight");
            if (recordOffset_ofDay % 100 == 0){
                wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". ", "recordOffset_ofDay: ", recordOffset_ofDay, ". ", "kernel 3 debug after sort");
                Check(cudaMemcpy(ph_perTimePoint_combinationMeasurementListing, pd_perTimePoint_combinationMeasurementListing, perTimePoint_combinationMeasurementByteSize, cudaMemcpyDeviceToHost));
                Check(cudaMemcpy(ph_perTimePoint_combinationIndexListing, pd_perTimePoint_combinationIndexListing, perTimePoint_combinationNr * sizeof(int), cudaMemcpyDeviceToHost));
                for (int i = 0; i < perTimePoint_combinationNr; i++){
                    cout << ph_perTimePoint_combinationIndexListing[i] << "\t" << ph_perTimePoint_combinationMeasurementListing[i] << "\n";
                }
            }
#endif
#if Debug_weightOfCombinationNIndex_txt
            if (recordOffset_ofDay % 100 == 0){
                wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". recordOffset_ofDay: ", recordOffset_ofDay, ". kernel 3 debug weight before update. flowIsGoingDown: ", flowIsGoingDown, ". flowIndicatorIndex : ", flowIndicatorIndex);
                Check(cudaMemcpy(ph_weightListing, pd_weightListing, weightNr * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < weightNr; i++){
                    cout << ph_weightListing[i] << "\t";
                    if (i > 0 && (i + 1) % perPredictStepLength_combinationNr == 0)
                        cout << "\n";
                }
                cout << "\n";
            }
#endif
            // call kernel 3
            if (queryRecordOkForWeight){
                updatePerTimePointRankToWeightListing <<< DimGrid_3, DimBlock_3 >>> (
                    pd_perTimePoint_combinationMeasurementListing,
                    pd_perTimePoint_combinationIndexListing,
                    pd_weightListing,
                    flowIsGoingDown,
                    flowIndicatorIndex,
                    perPredictStepLength_combinationNr,
                    perTimePoint_combinationNr);
                Check(cudaDeviceSynchronize());
            }
#if Debug_weightOfCombinationNIndex_txt
            else
                wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". ", "recordOffset_ofDay: ", recordOffset_ofDay, ". ", "kernel 3 debug skipping, not ok for weight");
            if (recordOffset_ofDay % 100 == 0){
                wbLog(DEBUG, "queryDayIndex_inAllGlobalDays: ", queryDayIndex_inAllGlobalDays, ". recordOffset_ofDay: ", recordOffset_ofDay, ". kernel 3 debug weight after update. flowIsGoingDown: ", flowIsGoingDown, ". flowIndicatorIndex : ", flowIndicatorIndex);
                Check(cudaMemcpy(ph_weightListing, pd_weightListing, weightNr * sizeof(float), cudaMemcpyDeviceToHost));
                for (int i = 0; i < weightNr; i++){
                    cout << ph_weightListing[i] << "\t";
                    if (i > 0 && (i + 1) % perPredictStepLength_combinationNr == 0)
                        cout << "\n";
                }
                cout << "\n";
            }
#endif
#endif /* if else WeightExists_isTesting */
#endif /* if Do_kernel */
        }/* for record of day end */
#if Debug_detail_timer
        if (queryDayIndex_inAllGlobalDays % 10 == 0){
            wbTime_stop(Generic, "One part: 10-day computation");
        }
#endif
    }/* for day of history end */
    
/* step 4: normalize */
#if !WeightExists_isTesting && !NormalizeWeightOnly
    wbLog(WARN, "Not NormalizeWeightOnly, but using Calculated Weights, not reading raw weights from HDD.");
    Check(cudaMemcpy(ph_weightListing, pd_weightListing, weightNr * sizeof(float), cudaMemcpyDeviceToHost));
    ofstream stream_weightBeforeNormalization(p_char_weightBeforeNormalizationFile, ios_base::binary);
    if (stream_weightBeforeNormalization.good()){
        stream_weightBeforeNormalization.write((char *)ph_weightListing, weightNr * sizeof(float));
    }
    else{
        wbLog(ERROR, "!stream_weightBeforeNormalization.good");
    }
    stream_weightBeforeNormalization.close();
#endif
#if !WeightExists_isTesting || NormalizeWeightOnly
    wbTime_start(Compute, "Normalization");
#if Debug_kernel4_normalize
    wbLog(DEBUG, "kernel 4 debug, weight before normalize, before threshold");
    for (int i = 0; i < weightNr; i++){
        cout << ph_weightListing[i] << "\t";
        if (i > 0 && (i + 1) % perPredictStepLength_combinationNr == 0) cout << "\n";
    }
    cout << "\n";
#endif
    thrust::device_ptr<float> p_weightListing(pd_weightListing);
    if (NormalizeDiscardRatio > 0.0f){
        for (int segment_index_t = 0; segment_index_t < DailyUpDown_segment_nr * FlowIndicator_segment_nr * predictStepLengthNr; segment_index_t++){
            int offset_kernel4_t = segment_index_t * perPredictStepLength_combinationNr;
            float minValue = thrust::reduce(p_weightListing + offset_kernel4_t, p_weightListing + offset_kernel4_t + perPredictStepLength_combinationNr, Float_max, thrust::minimum<float>());
            float maxValue = thrust::reduce(p_weightListing + offset_kernel4_t, p_weightListing + offset_kernel4_t + perPredictStepLength_combinationNr, -1.0f, thrust::maximum<float>());
#if Debug_kernel4_normalize
            cout << "Debug Normalize segment_index_t: " << segment_index_t << ". minValue: " << minValue << ". maxValue: " << maxValue;
#endif
            if (minValue < 0) minValue = 0;
            if (maxValue < 0) minValue = 0;
            float normalizeThresholdMin = minValue + (maxValue - minValue) * NormalizeDiscardRatio;
#if Debug_kernel4_normalize
            cout << ". threshold: " << normalizeThresholdMin << endl;
#endif
            normalizeThresholdTreatment <<< DimGrid_4_division, DimBlock_4_division >>> (
                pd_weightListing + offset_kernel4_t,
                normalizeThresholdMin);
            Check(cudaDeviceSynchronize());
        }
    }

#if Debug_kernel4_normalize
    wbLog(DEBUG, "kernel 4 debug, weight before normalize, after threshold");
    Check(cudaMemcpy(ph_weightListing, pd_weightListing, weightNr * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < weightNr; i++){
        cout << ph_weightListing[i] << "\t";
        if (i > 0 && (i + 1) % perPredictStepLength_combinationNr == 0) cout << "\n";
    }
    cout << "\n";
#endif
    for (int segment_index_t = 0; segment_index_t < DailyUpDown_segment_nr * FlowIndicator_segment_nr * predictStepLengthNr; segment_index_t++){
        int offset_kernel4_t = segment_index_t * perPredictStepLength_combinationNr;
        float sumTotal = thrust::reduce(p_weightListing + offset_kernel4_t, p_weightListing + offset_kernel4_t + perPredictStepLength_combinationNr);
        if (sumTotal < 1 && Score_valueMethod < 5){
            wbLog(WARN, "sumTotal too small, < 1.");
        }
        if (sumTotal < 0.1 && Score_valueMethod >= 5){
            wbLog(WARN, "sumTotal too small, < 0.1.");
        }

#if Debug_kernel4_normalize
        wbLog(DEBUG, "sumTotal: ", sumTotal);
#endif
        normalizeDivision <<< DimGrid_4_division, DimBlock_4_division >>> (pd_weightListing + offset_kernel4_t, sumTotal, numeric_limits<float>::epsilon());
        Check(cudaDeviceSynchronize());
    }
#if Debug_kernel4_normalize
    wbLog(DEBUG, "kernel 4 debug, weight after normalize, after threshold");
    Check(cudaMemcpy(ph_weightListing, pd_weightListing, weightNr * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < weightNr; i++){
        cout << ph_weightListing[i] << "\t";
        if (i > 0 && (i + 1) % perPredictStepLength_combinationNr == 0)
            cout << "\n";
    }
    cout << "\n";
#endif
    wbTime_stop(Compute, "Normalization");
#endif // !WeightExists_isTesting
    /* step 4: normalize - end */


    wbTime_stop(GPU, "Main GPU computation");
    //@@ Launch GPU Kernels - end

    //@@ Copy GPU memory back to CPU
    wbTime_start(Copy, "Copying to Host");
#if Debug_logic && Write_ph_output
    for (int i = 0; i < 10; i++){
        ph_output[i] = ph_input[i];
    }
#elif !Debug_logic
#if Write_ph_output
    Check(cudaMemcpy(ph_output, pd_output, outputByteSize, cudaMemcpyDeviceToHost));
#endif
#endif
#if Debug_sortedKnnPredict
    Check(cudaMemcpy(ph_sortedKnnPrediction, pd_sortedKnnPrediction, sortedKnnPredictionByteSize, cudaMemcpyDeviceToHost));
#endif
#if !WeightExists_isTesting && Do_kernel || NormalizeWeightOnly
Check(cudaMemcpy(ph_weightListing, pd_weightListing, weightNr * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying to Host");
#endif
    //@@ Copy GPU memory back to CPU - end

    //@@ Save result to disk files
    wbTime_start(Copy, "Saving to disk");
#if Write_ph_output && Do_kernel
    // ph_output, c.a. 1GB * Debug_write_ph_output_ratio
    std::ofstream stream_outFile(p_char_outputBinFile, std::ios_base::binary);
    if (stream_outFile.good()){
        stream_outFile.write((char *)ph_output, outputByteSize * Debug_write_ph_output_ratio);
    }
    else{
        wbLog(ERROR, "!stream_outFile.good");
    }
    stream_outFile.close();
#endif
#if Write_analysisFlag
    // ph_recordOkForAnalysisFlagListing, ~500KB
    ofstream stream_analysisFlagFile(p_char_analysisFlagListingFile, ios_base::binary);
    if (stream_analysisFlagFile.good()){
        stream_analysisFlagFile.write((char *)ph_recordOkForAnalysisFlagListing, recordFlagNr * sizeof(int));
    }
    else{
        wbLog(ERROR, "!ph_recordOkForAnalysisFlagListing.good");
    }
    stream_analysisFlagFile.close();
#endif
#if WeightExists_isTesting && Do_kernel
    // ph_weightedPredictListing ~3.5MB
    ofstream stream_weightedPredictFile(p_char_weightedPredictFile, ios_base::binary);
    if (stream_weightedPredictFile.good()){
        stream_weightedPredictFile.write((char *)ph_weightedPredictListing, completeDayNr * Daily_record_nr * Data_variable_nr * predictStepLengthNr * sizeof(float));
    }
    else{
        wbLog(ERROR, "!stream_weightedPredictFile.good");
    }
    stream_weightedPredictFile.close();
#elif !WeightExists_isTesting && Do_kernel || NormalizeWeightOnly
    // ph_weightListing ~100KB
    ofstream stream_weightFile(p_char_weightNormalizedFile, ios_base::binary);
    if (stream_weightFile.good()){
        stream_weightFile.write((char *)ph_weightListing, weightNr * sizeof(float));
    }
    else{
        wbLog(ERROR, "!stream_weightFile.good");
    }
    stream_weightFile.close();
#endif
#if Debug_allDayDistance
    ofstream stream_outFile_debug_allDayDistance_file("../../gpu-r-proj/Debug_allDayDistance_file.bin", ios::out | ios::binary);
    if (stream_outFile_debug_allDayDistance_file.good()){
        stream_outFile_debug_allDayDistance_file.write((char *)ph_oneTimePoint_searchStepLength_searchShift_day_distance, oneTimePointAllNeighborDistanceNr * sizeof(float));
    }
    else{
        wbLog(ERROR, "!stream_outFile_debug_allDayDistance_file.good: Debug_allDayDistance_file.bin");
    }
    stream_outFile_debug_allDayDistance_file.close();
#endif
#if Debug_sortedKnnPredict
    ofstream stream_outFile_pd_sortedKnnPrediction("../../gpu-r-proj/Debug_sortedKnnPredict.bin", ios::out | ios::binary);
    if (stream_outFile_pd_sortedKnnPrediction.good()){
        stream_outFile_pd_sortedKnnPrediction.write((char *)ph_sortedKnnPrediction, sortedKnnPredictionByteSize);
    }
    else{
        wbLog(ERROR, "!stream_outFile_pd_sortedKnnPrediction.good: Debug_sortedKnnPredict.bin");
    }
    stream_outFile_pd_sortedKnnPrediction.close();
#endif
    wbTime_stop(Copy, "Saving to disk");
    //@@ Save result to disk files - end

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free GPU memory
    Check(cudaFree(pd_input));
#if Write_ph_output
    Check(cudaFree(pd_output));
#endif
    Check(cudaFree(pd_perTimePoint_combinationMeasurementListing));
    Check(cudaFree(pd_perTimePoint_combinationIndexListing));
    Check(cudaFree(pd_oneTimePoint_searchStepLength_searchShift_day_distance));
    Check(cudaFree(pd_oneTimePoint_searchStepLength_searchShift_day_index));
    Check(cudaFree(pd_searchStepLengthListing));
    Check(cudaFree(pd_windowSizeListing));
    Check(cudaFree(pd_kLevel1Listing));
    Check(cudaFree(pd_predictStepLengthListing));
    Check(cudaFree(pd_incidentFlagScannedListing));
#if Debug_weightedPredictDistance
    Check(cudaFree(pd_unweightedPredictListing));
#endif
    Check(cudaFree(pd_weightListing));
    wbTime_stop(GPU, "Freeing GPU Memory");
    //@@ Free GPU memory - end

    //@@ Free host memory
    wbTime_start(GPU, "Freeing Host Memory");
    mxDestroyArray(pMxArray);
    if (matClose(pMatFile) != 0) { printf("Error closing p_char_inputMatFile %s\n", p_char_inputMatFile); return(1); }
    //free(ph_input); // illegal memory free, already done in mxDestroyArray().
#if Write_ph_output
    free(ph_output);
#endif
    free(ph_incidentFlagScannedListing);
    free(ph_holidayFlagListing);
    free(ph_recordOkForAnalysisFlagListing);
    free(ph_weightListing);
#if Debug_weightedPredictDistance
    free(ph_unweightedPredictListing);
#endif
#if WeightExists_isTesting
    free(ph_weightedPredictListing);
#elif !WeightExists_isTesting
#endif
#if Debug_allDayDistance
    free(ph_oneTimePoint_searchStepLength_searchShift_day_distance);
#endif
#if Debug_sortedKnnPredict
    free(ph_oneTimePoint_searchStepLength_searchShift_day_index);
#endif
    wbTime_stop(GPU, "Freeing Host Memory");
    //@@ Free host memory - end

    printDateNTime();
    return 0;
}
