#ifndef IndINCLUDE
#define IndINCLUDE
#include "Header.h"
#endif

using namespace std;

int main(int argc, char * argv[]){

        // ---- CLI parameters and global settings (clean copy: wht 24/30 only) ----
        ofstream fileRESULTS;

        IloEnv env;

        if (argc != 15) {
            cout << "Usage:" << endl;
            cout << "  " << argv[0]
                 << " seed N_sc N_pb N_tp N_it N_pn beta lost al alphaScale wht"
                 << " FixMode DynamicFix evalMode" << endl;
            cout << "Supported wht:" << endl;
            cout << "  24 30" << endl;
            return 1;
        }

        int arg = 1;
        int seed = std::atoi(argv[arg++]);
        N_sc = std::atoi(argv[arg++]); //Number of demand scenarios (xi)
        N_pb = std::atoi(argv[arg++]); //Number of probing scenarios (eta)
        N_tp = std::atoi(argv[arg++]); //Number of time periods
        N_it = std::atoi(argv[arg++]); //Number of items
        N_pn = std::atoi(argv[arg++]); //Number of probing components
        double beta = std::atof(argv[arg++]); //Capacity level
        lost = std::atof(argv[arg++]);      //Penalty cost for lost sales
        al = std::atof(argv[arg++]);        //probing cost scaling factor (for ProbingCost_generation)
        alphaScale = std::atof(argv[arg++]); // multiplier for probing costs
        int wht = std::atoi(argv[arg++]);  //Model to solve the problem
        if (wht != 24 && wht != 30) {
            cout << "Unsupported wht in this clean copy. Use 24 or 30." << endl;
            return 1;
        }

        const char* outFile = RESULTS;
        if (wht == 24) {
            outFile = "./Results_BB_New.txt";
        } else if (wht == 30) {
            outFile = "./Results_BB_AIR.txt";
        }
        bool writeHeader = false;
        {
            std::ifstream existing(outFile, std::ios::in);
            writeHeader = (!existing.good() || existing.peek() == std::ifstream::traits_type::eof());
        }
        fileRESULTS.open(outFile, ios::app);
        ActiveWht = wht;

        // Defaults
        mu_D = 100.0;     // Expected demand (default)
        rho = -0.7;       // Correlation factor (default)
        CorrMode = 1;     // W_it_pn mode when N_it==N_pn (0=single-component, 1=multi-component)
        int bbMode = 0;   // 0=single-element, 1=multi-element
        int bbCache = 1;  // 0=disable caching, 1=enable caching (hashing)
        int evalMode = 0; // 0=joint, 1=item-wise
        FixMode = 0; // 0=none, 1=static props3-5, 2=node-residual props3-5 (wht24), 3=static+node-residual
        DynamicFix = 0; // 0=off, 1=on (dynamic fixing in Ma-style branch-and-bound)
        HeurFreeWindow = 1; //time window for heuristics (for wht 24)
        HeurLBMode = 1; // 0=valid Lower, 1=heuristic Lowerbound (for wht 24)
        HeurRepairMode = 0; // 0=off(current),1=repair,2=repair+LP-polish,3=alternating,4=relaxed-Y strengthened LP,5=alternating+short intensification MIP
        HeurRepairV3UpdateMode = 0; // 0=parallel(Jacobi),1=sequential(Gauss-Seidel)
        HeurRepairV3MaxIter = 20; // max alternations for HeurRepairMode=3/4/5
        HeurIntensifyTiLim = 10.0; // short MIP time limit for HeurRepairMode=5
        Wht24TimeLimitSec = 3600.0; // total algorithm time limit for wht 24
        TauPackingLBMode = 0; // 0=off, 1=on (tau-packing LB strengthening)
        ApproxGapEps = 0.05; // acceptance tolerance for AIR-B&B
        ApproxReuseMode = 1; // 0=off, 1=on
        ApproxLocalImproveMode = 1; // 0=off, 1=on
        ApproxLBMode = 1; // 0=base LR/LP, 1=base LR/LP + packing LB
        ApproxGlobalLagLBMode = 1; // 0=off, 1=on
        ApproxGlobalLagMaxIter = 1; // LR subgradient iterations for global LB
        ApproxGlobalLagStep0 = 1.0; // LR step size for global LB
        ApproxExactFallbackMode = 1; // 0=off, 1=on
        SetupCertEnable = 0; // legacy setup certification disabled in this clean copy.
        SetupCertMode = 0;
        SetupCertTau = 0.0;
        SetupCertLagMaxIter = 1;
        SetupCertLagStep0 = 1.0;
        SetupCertItemTiLim = 30.0;
        DebugTargetMask = -1; // optional debug mask for branch-and-bound tracing

        FixMode = std::atoi(argv[arg++]);
        DynamicFix = std::atoi(argv[arg++]);
        evalMode = std::atoi(argv[arg++]);

        if (FixMode < 0 || FixMode > 3) {
            cout << "ERROR: FixMode must be in [0,3]." << endl;
            return 1;
        }
        if (DynamicFix < 0 || DynamicFix > 1) {
            cout << "ERROR: DynamicFix must be in [0,1]." << endl;
            return 1;
        }
        if (evalMode < 0 || evalMode > 1) {
            cout << "ERROR: evalMode must be in [0,1]." << endl;
            return 1;
        }
        if (wht == 30 && evalMode != 1) {
            cout << "WARNING: wht=30 AIR-B&B is intended to run with evalMode=1." << endl;
        }

        bool rhoPnProvided = false;
        RhoPn.assign(N_pn, rho);

        VI_pro1 = 0;
        VI_pro2 = 0;
        VI_pro3 = 0;
        VI_routine = 0;
        cout <<"######## Parameters ########"<< endl; 
        cout << "Seed: " << seed << endl;
        cout << "Number of time periods: " << N_tp << endl;
        cout << "Number of demand scenarios: " << N_sc << endl;
        cout << "Number of probing scenarios: " << N_pb << endl;
        cout << "Number of items: " << N_it << endl;
        cout << "Number of probing: " << N_pn << endl;
        cout << "Capacity level (beta): " << beta << endl;
        cout << "Expected demand (mu_D): " << mu_D << endl;
        cout << "probing cost scaling factor (al): " << al << endl;
        cout << "probing cost multiplier (alphaScale): " << alphaScale << endl;
        cout << "Lost sales penalty (lost): " << lost << endl;
        cout << "W_it_pn mode (only affects N_it==N_pn): " << CorrMode
             << " (0=single-component, 1=multi-component)" << endl;
        cout << "Model to solve (wht): " << wht << endl;
        cout << "FixMode (props): " << FixMode << endl;
        cout << "DynamicFix (B&B Ma): " << DynamicFix << endl;
        cout << "EvalMode (0=joint,1=item): " << evalMode << endl;
        if (wht == 24) {
            cout << "Heuristic free window (Y): " << HeurFreeWindow << endl;
            cout << "Heuristic LB mode (0=valid,1=heur): " << HeurLBMode << endl;
            cout << "Heuristic repair mode (0=off,1=repair,2=repair+LP,3=alternating,4=relaxed-Y,5=alt+shortMIP): " << HeurRepairMode << endl;
            cout << "Heuristic V3 update mode (0=parallel,1=sequential): " << HeurRepairV3UpdateMode << endl;
            cout << "Heuristic V3 max iterations: " << HeurRepairV3MaxIter << endl;
            cout << "Heuristic intensification TiLim (s): " << HeurIntensifyTiLim << endl;
            cout << "wht24 total time limit (s): " << Wht24TimeLimitSec << endl;
            cout << "Tau-packing LB mode (0=off,1=on): " << TauPackingLBMode << endl;
        }
        if (wht == 30) {
            cout << "AIR gap epsilon: " << ApproxGapEps << endl;
            cout << "AIR reuse mode (0=off,1=on): " << ApproxReuseMode << endl;
            cout << "AIR local improvement mode (0=off,1=on): " << ApproxLocalImproveMode << endl;
            cout << "AIR LB mode (0=base,1=base+packing): " << ApproxLBMode << endl;
            cout << "AIR global lagrangian LB mode (0=off,1=on): " << ApproxGlobalLagLBMode << endl;
            cout << "AIR global lagrangian max iterations: " << ApproxGlobalLagMaxIter << endl;
            cout << "AIR global lagrangian step0: " << ApproxGlobalLagStep0 << endl;
            cout << "AIR exact fallback mode (0=off,1=on): " << ApproxExactFallbackMode << endl;
        }

        // ---- Random seed and data generation ----
        // Fix seed for instance generation
        srand(seed);
        std::mt19937 gen(seed);
        if (!rhoPnProvided) {
            std::uniform_real_distribution<double> rhoDist(-0.7, 0.7);
            double sumRho = 0.0;
            for (int i = 0; i < N_pn; ++i) {
                RhoPn[i] = rhoDist(gen);
                sumRho += RhoPn[i];
            }
            rho = (N_pn > 0) ? (sumRho / static_cast<double>(N_pn)) : rho;
        }

        //Create vector of solution to iterative update solutions value throughout the decomoposition algorithm
        solution_vector(env);
        //Instance generation
        Instance_Generation(env, gen, beta);

        VIPathCount = 0;
        VITreeCount = 0;
        BBNodesExplored = 0;
        BBDynFixedCount = 0;
        BBPruned = 0;
        CacheMaskHit = 0;
        CacheTauHit = 0;
        AirTauSolvedCount = 0;
        AirTauAcceptedBaseCount = 0;
        AirTauAcceptedLocalCount = 0;
        AirTauEnteredRefineCount = 0;
        AirTauEnteredLagCount = 0;
        AirTauAcceptedAfterLagCount = 0;
        AirTauAcceptedAfterReuseImproveCount = 0;
        AirTauAcceptedAfterSeedImproveCount = 0;
        AirTauExactFallbackCount = 0;
        AirTauExactGapStoppedCount = 0;
        AirTauExactOptimalCount = 0;
        BestZCount = -1;
        WhtLPRelax = 0.0;
        WhtLPRelaxReinf = 0.0;
        WhtRootLB = 0.0;
        WhtBestLB = 0.0;
        WhtBestUB = 0.0;
        FixCount = 0;

        if ((wht == 24 || wht == 30) && (FixMode == 1 || FixMode == 3)) {
            RunFixingPropositions(env);
        } else {
            FixZ.assign(N_pn, -1);
        }

        // ---- Model selection (clean copy: wht 24/30 only) ----
        // Vector to export solution values
        IloNumArray Solution (env);

        //time algorithms
        double Time_GlobalModel;
        double Time_Decomp_1=0;
        double Time_Decomp_2=0;
        double Time_Decomp_3=0;

        struct timespec start_GM,end_GM;
        clock_gettime(CLOCK_MONOTONIC,&start_GM);

        if(wht==24){
            cout << "#################### Ma-style enhanced branch-and-bound ########################"<< endl;
            Branch_and_Bound_Ma(env, Solution, bbMode, evalMode, bbCache);
            cout << "############################################"<< endl;
        }
        else if(wht==30){
            cout << "#################### AIR-B&B ########################"<< endl;
            Approx_Branch_and_Bound(env, Solution, bbMode, evalMode, bbCache);
            cout << "############################################"<< endl;
        }
        else{
            cout << "Unsupported wht in this clean copy. Use 24 or 30." << endl;
            return 1;
        }


        clock_gettime(CLOCK_MONOTONIC,&end_GM);
        Time_GlobalModel = BILLION*(end_GM.tv_sec - start_GM.tv_sec)+end_GM.tv_nsec -start_GM.tv_nsec;
        Time_GlobalModel=Time_GlobalModel/pow((double)10,9);
        cout << "Time New formulation = " << Time_GlobalModel << endl;
        if (wht == 30) {
            cout << "AIR tau solved = " << AirTauSolvedCount << endl;
            cout << "AIR tau accepted by gap (base) = " << AirTauAcceptedBaseCount << endl;
            cout << "AIR tau entered refinement = " << AirTauEnteredRefineCount << endl;
            cout << "AIR tau entered lagrangian refinement = " << AirTauEnteredLagCount << endl;
            cout << "AIR tau accepted after refinement = " << AirTauAcceptedLocalCount << endl;
            cout << "AIR tau accepted after lagrangian = " << AirTauAcceptedAfterLagCount << endl;
            cout << "AIR tau accepted after reuse intensification = " << AirTauAcceptedAfterReuseImproveCount << endl;
            cout << "AIR tau accepted after seed improvement = " << AirTauAcceptedAfterSeedImproveCount << endl;
            cout << "AIR tau sent to exact fallback = " << AirTauExactFallbackCount << endl;
            cout << "AIR tau stopped exact by gap = " << AirTauExactGapStoppedCount << endl;
            cout << "AIR tau solved to optimality = " << AirTauExactOptimalCount << endl;
        }

        // Solution.add(Time_Decomp_1);
        // Solution.add(Time_Decomp_2);
        // Solution.add(Time_GlobalModel);

        int nodesExploredOut = 0;
        long long cutsAddedOut = 0;
        int fixCountOut = 0;
        int dynFixOut = 0;
        int prunedOut = 0;
        long long cacheMaskHitOut = 0;
        long long cacheTauHitOut = 0;
        long long airTauSolvedOut = 0;
        long long airTauAcceptedBaseOut = 0;
        long long airTauAcceptedLocalOut = 0;
        long long airTauEnteredRefineOut = 0;
        long long airTauEnteredLagOut = 0;
        long long airTauAcceptedAfterLagOut = 0;
        long long airTauAcceptedAfterReuseImproveOut = 0;
        long long airTauAcceptedAfterSeedImproveOut = 0;
        long long airTauExactFallbackOut = 0;
        long long airTauExactGapStoppedOut = 0;
        long long airTauExactOptimalOut = 0;
        double bestLBOut = 0.0;
        double bestUBOut = 0.0;
        if (wht == 24 || wht == 30) {
            nodesExploredOut = BBNodesExplored;
            cutsAddedOut = VIPathCount + VITreeCount;
            fixCountOut = FixCount;
            dynFixOut = BBDynFixedCount;
        }
        if (wht == 24 || wht == 30) {
            prunedOut = BBPruned;
            cacheMaskHitOut = CacheMaskHit;
            cacheTauHitOut = CacheTauHit;
            airTauSolvedOut = AirTauSolvedCount;
            airTauAcceptedBaseOut = AirTauAcceptedBaseCount;
            airTauAcceptedLocalOut = AirTauAcceptedLocalCount;
            airTauEnteredRefineOut = AirTauEnteredRefineCount;
            airTauEnteredLagOut = AirTauEnteredLagCount;
            airTauAcceptedAfterLagOut = AirTauAcceptedAfterLagCount;
            airTauAcceptedAfterReuseImproveOut = AirTauAcceptedAfterReuseImproveCount;
            airTauAcceptedAfterSeedImproveOut = AirTauAcceptedAfterSeedImproveCount;
            airTauExactFallbackOut = AirTauExactFallbackCount;
            airTauExactGapStoppedOut = AirTauExactGapStoppedCount;
            airTauExactOptimalOut = AirTauExactOptimalCount;
            if (Solution.getSize() >= 2) {
                bestLBOut = Solution[0];
                bestUBOut = Solution[1];
            }
        }
        const bool isAirBranchBound = (wht == 30);
        const bool isActiveBranchBound = (wht == 24 || wht == 30);
        if (isAirBranchBound) {
            if (writeHeader) {
                fileRESULTS
                    << "seed\tN_tp\tN_sc\tN_pb\tN_it\tN_pn\tbeta\tmu_D\tlost\trho\twht\t"
                    << "al\talphaScale\tFixMode\tDynamicFix\tevalMode\t"
                    << "HeurFreeWindow\tHeurLBMode\tHeurRepairMode\tHeurRepairV3UpdateMode\tHeurRepairV3MaxIter\t"
                    << "HeurIntensifyTiLim\tWht24TimeLimitSec\tTauPackingLBMode\t"
                    << "ApproxGapEps\tApproxReuseMode\tApproxLocalImproveMode\tApproxLBMode\t"
                    << "ApproxGlobalLagLBMode\tApproxGlobalLagMaxIter\tApproxGlobalLagStep0\tApproxExactFallbackMode\t"
                    << "bestLB\tbestUB\tTime\tNodes\tCuts\tFixCount\tDynFix\tPruned\tCacheMaskHit\tCacheTauHit\tBestZCount\t"
                    << "AirTauSolved\tAirTauAcceptedBase\tAirTauEnteredRefine\tAirTauEnteredLag\t"
                    << "AirTauAcceptedAfterRefinement\tAirTauAcceptedAfterLag\tAirTauAcceptedAfterReuseImprove\t"
                    << "AirTauAcceptedAfterSeedImprove\tAirTauExactFallback\tAirTauExactGapStopped\tAirTauExactOptimal"
                    << endl;
            }
            fileRESULTS
                << seed << "\t"
                << N_tp << "\t"
                << N_sc << "\t"
                << N_pb << "\t"
                << N_it << "\t"
                << N_pn << "\t"
                << beta << "\t"
                << mu_D << "\t"
                << lost << "\t"
                << rho << "\t"
                << wht << "\t"
                << al << "\t"
                << alphaScale << "\t"
                << FixMode << "\t"
                << DynamicFix << "\t"
                << evalMode << "\t"
                << HeurFreeWindow << "\t"
                << HeurLBMode << "\t"
                << HeurRepairMode << "\t"
                << HeurRepairV3UpdateMode << "\t"
                << HeurRepairV3MaxIter << "\t"
                << HeurIntensifyTiLim << "\t"
                << Wht24TimeLimitSec << "\t"
                << TauPackingLBMode << "\t"
                << ApproxGapEps << "\t"
                << ApproxReuseMode << "\t"
                << ApproxLocalImproveMode << "\t"
                << ApproxLBMode << "\t"
                << ApproxGlobalLagLBMode << "\t"
                << ApproxGlobalLagMaxIter << "\t"
                << ApproxGlobalLagStep0 << "\t"
                << ApproxExactFallbackMode << "\t"
                << bestLBOut << "\t"
                << bestUBOut << "\t"
                << Time_GlobalModel << "\t"
                << nodesExploredOut << "\t"
                << cutsAddedOut << "\t"
                << fixCountOut << "\t"
                << dynFixOut << "\t"
                << prunedOut << "\t"
                << cacheMaskHitOut << "\t"
                << cacheTauHitOut << "\t"
                << BestZCount << "\t"
                << airTauSolvedOut << "\t"
                << airTauAcceptedBaseOut << "\t"
                << airTauEnteredRefineOut << "\t"
                << airTauEnteredLagOut << "\t"
                << airTauAcceptedLocalOut << "\t"
                << airTauAcceptedAfterLagOut << "\t"
                << airTauAcceptedAfterReuseImproveOut << "\t"
                << airTauAcceptedAfterSeedImproveOut << "\t"
                << airTauExactFallbackOut << "\t"
                << airTauExactGapStoppedOut << "\t"
                << airTauExactOptimalOut
                << endl;
        } else if (isActiveBranchBound) {
            if (writeHeader) {
                fileRESULTS
                    << "seed\tN_tp\tN_sc\tN_pb\tN_it\tN_pn\tbeta\tmu_D\tlost\trho\twht\t"
                    << "al\talphaScale\tFixMode\tDynamicFix\tevalMode\t"
                    << "HeurFreeWindow\tHeurLBMode\tHeurRepairMode\tHeurRepairV3UpdateMode\tHeurRepairV3MaxIter\t"
                    << "HeurIntensifyTiLim\tWht24TimeLimitSec\tTauPackingLBMode\t"
                    << "bestLB\tbestUB\tTime\tNodes\tCuts\tFixCount\tDynFix\tPruned\tCacheMaskHit\tCacheTauHit\tBestZCount"
                    << endl;
            }
            fileRESULTS
                << seed << "\t"
                << N_tp << "\t"
                << N_sc << "\t"
                << N_pb << "\t"
                << N_it << "\t"
                << N_pn << "\t"
                << beta << "\t"
                << mu_D << "\t"
                << lost << "\t"
                << rho << "\t"
                << wht << "\t"
                << al << "\t"
                << alphaScale << "\t"
                << FixMode << "\t"
                << DynamicFix << "\t"
                << evalMode << "\t"
                << HeurFreeWindow << "\t"
                << HeurLBMode << "\t"
                << HeurRepairMode << "\t"
                << HeurRepairV3UpdateMode << "\t"
                << HeurRepairV3MaxIter << "\t"
                << HeurIntensifyTiLim << "\t"
                << Wht24TimeLimitSec << "\t"
                << TauPackingLBMode << "\t"
                << bestLBOut << "\t"
                << bestUBOut << "\t"
                << Time_GlobalModel << "\t"
                << nodesExploredOut << "\t"
                << cutsAddedOut << "\t"
                << fixCountOut << "\t"
                << dynFixOut << "\t"
                << prunedOut << "\t"
                << cacheMaskHitOut << "\t"
                << cacheTauHitOut << "\t"
                << BestZCount
                << endl;
        }
        fileRESULTS.close();

}   
