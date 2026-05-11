//For using some specific functions in C++
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm> // For std::find
#include <unordered_set>
#include <random>
#include <unordered_map>
#include <thread>

//For using standard libraries
#include <ilcplex/ilocplex.h>
using namespace std;

ILOSTLBEGIN

#define BILLION 1000000000L

//Definition of arrays of decision variables
typedef IloArray <IloNumVarArray> IloNumVarArray2;
typedef IloArray <IloNumVarArray2> IloNumVarArray3;
typedef IloArray <IloNumVarArray3> IloNumVarArray4;
typedef IloArray <IloBoolVarArray> IloBoolVarArray2;
typedef IloArray <IloBoolVarArray2> IloBoolVarArray3; 
typedef IloArray <IloExprArray> IloExprArray2;
typedef IloArray <IloExprArray2> IloExprArray3;
typedef IloArray <IloModel> IloModelArray;
typedef IloArray <IloObjective> IloObjectiveArray ;
typedef IloArray <IloNumArray2> IloNumArray3;
typedef IloArray <IloNumArray3> IloNumArray4;

extern int Cap;
extern int N_tp;
extern int N_sc;
extern int N_pb;
extern int N_it;
extern int N_pn;
extern int VI_pro1;
extern int VI_pro2;
extern int VI_pro3;
extern int VI_routine;
extern double mu_D;
extern double rho;
extern std::vector<double> RhoPn;
extern double lost;
extern double al;
extern double alphaScale;
extern double EtaRoundingScale;
extern int FixMode;
extern int DynamicFix;
extern int HeurFreeWindow;
extern int HeurLBMode;
extern int HeurRepairMode;
extern int HeurRepairV3UpdateMode;
extern int HeurRepairV3MaxIter;
extern double HeurIntensifyTiLim;
extern double Wht24TimeLimitSec;
extern int TauPackingLBMode;
extern double ApproxGapEps;
extern int ApproxReuseMode;
extern int ApproxLocalImproveMode;
extern int ApproxLBMode;
extern int ApproxGlobalLagLBMode;
extern int ApproxGlobalLagMaxIter;
extern double ApproxGlobalLagStep0;
extern int ApproxExactFallbackMode;
extern int SetupCertEnable;
extern int SetupCertMode;
extern double SetupCertTau;
extern int SetupCertLagMaxIter;
extern double SetupCertLagStep0;
extern double SetupCertItemTiLim;
extern int DebugTargetMask;
extern int ActiveWht;
extern std::vector<int> FixZ;
extern int FixCount;
extern int BBNodesExplored;
extern int BBDynFixedCount;
extern int BBPruned;
extern long long CacheMaskHit;
extern long long CacheTauHit;
extern long long AirTauSolvedCount;
extern long long AirTauAcceptedBaseCount;
extern long long AirTauAcceptedLocalCount;
extern long long AirTauEnteredRefineCount;
extern long long AirTauEnteredLagCount;
extern long long AirTauAcceptedAfterLagCount;
extern long long AirTauAcceptedAfterReuseImproveCount;
extern long long AirTauAcceptedAfterSeedImproveCount;
extern long long AirTauExactFallbackCount;
extern long long AirTauExactGapStoppedCount;
extern long long AirTauExactOptimalCount;
extern long long SetupCertCheckCount;
extern long long SetupCertFullMipCount;
extern long long SetupCertApproxAcceptedCount;
extern long long SetupCertCoverageCounts[11];
extern double SetupCertCoverageUbSums[11];
extern double SetupCertCoverageLbSums[11];
extern long long SetupCertFiniteLBCount;
extern double SetupCertFiniteUBSum;
extern double SetupCertFiniteLBSum;
extern long long SetupCertBestSolLBCount;
extern double SetupCertBestSolUBSum;
extern double SetupCertBestSolLBSum;
extern long long SetupCertGlobalLBCount;
extern double SetupCertGlobalLBSum;
extern long long SetupCertIncCompLBCount;
extern double SetupCertIncCompLBSum;
extern long long SetupCertMaxLpCount;
extern long long SetupCertMaxPackCount;
extern long long SetupCertMaxLpStrongCount;
extern long long SetupCertMaxLagCount;
extern int BestZCount;
extern double WhtLPRelax;
extern double WhtLPRelaxReinf;
extern double WhtRootLB;
extern double WhtBestLB;
extern double WhtBestUB;
extern long long VIPathCount;
extern long long VITreeCount;

// ---- Indistinguishability / kappa(z,gamma)=tau mapping ----
extern int N_z;  // number of probing vectors = 2^{N_it}
extern std::vector<std::vector<int>> KappaTau;
extern std::vector<std::vector<int>> TauClassId;
extern std::vector<std::vector<int>> TauReps;
void BuildKappaTauMappings(double roundingScale = 1e6);

//Definition for the character stand for abbrevation on input + result files
extern char  INPUT [500] ;
extern char  RESULTS [500] ;

// - problem parameters
extern IloNumArray  alpha;
extern IloNumArray2 f;
extern IloNumArray2 p;
extern IloNumArray2 h;
extern IloNumArray2 b;
extern IloNumArray2 eta;
extern IloNumArray4 d;
extern IloNumArray3 d_xi;
extern IloNumArray2 Prob_xi_given_eta;

extern int CorrMode;

extern IloNumArray2 BigM_np;
extern IloNumArray3 BigM;

extern IloNumArray2 SolY_item;
extern IloNumArray2 SolX_item;
extern IloNumArray3 SolI_item;
extern IloNumArray3 SolL_item;

extern IloNumArray3 SolY_prob;
extern IloNumArray3 SolX_prob;
extern IloNumArray4 SolI_prob;
extern IloNumArray4 SolL_prob;

void Instance_Generation (IloEnv env, std::mt19937& gen, double beta);
void RunFixingPropositions(IloEnv env);
void solution_vector (IloEnv env);

double Branch_and_Bound_Ma (IloEnv env, IloNumArray Solution, int bbMode, int evalMode, int bbCache);
double Approx_Branch_and_Bound(IloEnv env, IloNumArray Solution, int bbMode, int evalMode, int bbCache);
