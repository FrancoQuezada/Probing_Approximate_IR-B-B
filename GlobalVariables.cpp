#ifndef IndINCLUDE
#define IndINCLUDE
#include "Header.h"
#endif

// address of the file containing the input data 
char INPUT [500] = "./Instances_test/";
char RESULTS [500] = "./Results_article_R1_Value_of_item_specific_model.txt";


int Cap;
// int BigM;
int N_tp;
int N_sc;
int N_pb;
int N_it;
int N_pn;
int VI_pro1;
int VI_pro2;
int VI_pro3;
int VI_routine;
double mu_D;
double rho;
double lost;
double al;
double alphaScale;
double EtaRoundingScale;
int FixMode;
int DynamicFix;
int HeurFreeWindow = 1;
int HeurLBMode = 1;
int HeurRepairMode = 0;
int HeurRepairV3UpdateMode = 0;
int HeurRepairV3MaxIter = 20;
double HeurIntensifyTiLim = 10.0;
double Wht24TimeLimitSec = 3600.0;
int TauPackingLBMode = 0;
double ApproxGapEps = 0.05;
int ApproxReuseMode = 1;
int ApproxLocalImproveMode = 1;
int ApproxLBMode = 1;
int ApproxGlobalLagLBMode = 1;
int ApproxGlobalLagMaxIter = 1;
double ApproxGlobalLagStep0 = 1.0;
int ApproxExactFallbackMode = 1;
int SetupCertEnable = 0;
int SetupCertMode = 0;
double SetupCertTau = 0.0;
int SetupCertLagMaxIter = 1;
double SetupCertLagStep0 = 1.0;
double SetupCertItemTiLim = 30.0;
int DebugTargetMask;
int ActiveWht = -1;
std::vector<int> FixZ;
int FixCount;
int BBNodesExplored;
int BBDynFixedCount;
int BBPruned;
long long CacheMaskHit;
long long CacheTauHit;
long long AirTauSolvedCount;
long long AirTauAcceptedBaseCount;
long long AirTauAcceptedLocalCount;
long long AirTauEnteredRefineCount;
long long AirTauEnteredLagCount;
long long AirTauAcceptedAfterLagCount;
long long AirTauAcceptedAfterReuseImproveCount;
long long AirTauAcceptedAfterSeedImproveCount;
long long AirTauExactFallbackCount;
long long AirTauExactGapStoppedCount;
long long AirTauExactOptimalCount;
long long SetupCertCheckCount;
long long SetupCertFullMipCount;
long long SetupCertApproxAcceptedCount;
long long SetupCertCoverageCounts[11];
double SetupCertCoverageUbSums[11];
double SetupCertCoverageLbSums[11];
long long SetupCertFiniteLBCount;
double SetupCertFiniteUBSum;
double SetupCertFiniteLBSum;
long long SetupCertBestSolLBCount;
double SetupCertBestSolUBSum;
double SetupCertBestSolLBSum;
long long SetupCertGlobalLBCount;
double SetupCertGlobalLBSum;
long long SetupCertIncCompLBCount;
double SetupCertIncCompLBSum;
long long SetupCertMaxLpCount;
long long SetupCertMaxPackCount;
long long SetupCertMaxLpStrongCount;
long long SetupCertMaxLagCount;
int BestZCount;
double WhtRootLB;
double WhtBestLB;
double WhtBestUB;
double WhtLPRelax;
double WhtLPRelaxReinf;
long long VIPathCount;
long long VITreeCount;
std::vector<double> RhoPn;
std::vector<std::vector<double>> W_it_pn;
std::vector<std::vector<double>> RhoItPn;

int N_z;
std::vector<std::vector<int>> KappaTau;
std::vector<std::vector<int>> TauClassId;
std::vector<std::vector<int>> TauReps;


// - problem parameters
IloNumArray  alpha;
IloNumArray2 f;
IloNumArray2 p;
IloNumArray2 h;
IloNumArray2 b;
IloNumArray4 d;
IloNumArray2 eta;
IloNumArray3 d_xi;
IloNumArray2 Prob_xi_given_eta;

int CorrMode = 0;

IloNumArray2 SolY_item;
IloNumArray2 SolX_item;
IloNumArray3 SolI_item;
IloNumArray3 SolL_item;

IloNumArray3 SolY_prob;
IloNumArray3 SolX_prob;
IloNumArray4 SolI_prob;
IloNumArray4 SolL_prob;

IloNumArray2 BigM_np;
IloNumArray3 BigM;




double normal_dist(std::mt19937& gen, double mu, double sigma){
    // Random number generator
    // std::random_device rd;
    std::normal_distribution<> conditional_dist(mu, sigma);
    int val = conditional_dist(gen);
    return val;
}


// Initialize solution arrays used by the algorithms.
void solution_vector (IloEnv env){


    SolX_item = IloNumArray2 (env, N_it);
    SolY_item = IloNumArray2 (env, N_it);

    SolL_item = IloNumArray3 (env, N_it);
    SolI_item = IloNumArray3 (env, N_it);

    SolY_prob = IloNumArray3 (env, N_it);
    SolX_prob = IloNumArray3 (env, N_it);

    SolL_prob = IloNumArray4 (env, N_it);
    SolI_prob = IloNumArray4 (env, N_it);

    for (int it = 0; it < N_it; ++it) {
        SolX_item[it] = IloNumArray (env, N_tp+1);
        SolY_item[it] = IloNumArray (env, N_tp+1);

        SolL_item[it] = IloNumArray2 (env, N_sc);
        SolI_item[it] = IloNumArray2 (env, N_sc);

        SolX_prob[it] = IloNumArray2 (env, N_pb);
        SolY_prob[it] = IloNumArray2 (env, N_pb);

        SolL_prob[it] = IloNumArray3 (env, N_sc);
        SolI_prob[it] = IloNumArray3 (env, N_sc);

        for (int pb = 0; pb < N_pb; ++pb) {
            SolX_prob[it][pb] = IloNumArray (env, N_tp+1);
            SolY_prob[it][pb] = IloNumArray (env, N_tp+1);
        }

        for (int sc = 0; sc < N_sc; sc++) {
            SolI_prob[it][sc] = IloNumArray2 (env, N_pb);
            SolL_prob[it][sc] = IloNumArray2 (env, N_pb);
            for (int pb = 0; pb < N_pb; ++pb) {
                SolI_prob[it][sc][pb] = IloNumArray (env, N_tp+1);
                SolL_prob[it][sc][pb] = IloNumArray (env, N_tp+1);
            }
        }
    }
}

// Generate eta and xi scenarios (d) and initialize core arrays.
void scenarios_generation(IloEnv env, std::mt19937& gen, IloNum& Expdem, IloNumArray& ExpDemItem,
                          double& mu_P, double& sigma_P, double& sigma_D){

    //Parameters definition
    f = IloNumArray2 (env, N_tp+1);
    p = IloNumArray2 (env, N_tp+1);
    h = IloNumArray2 (env, N_tp+1);
    b = IloNumArray2 (env, N_tp+1);
    d = IloNumArray4 (env, N_tp+1);
    eta = IloNumArray2 (env, N_pn);

    sigma_P = 2;     // Std deviation of Price
    mu_P = 50;       // Mean of Price

    for(int it=0; it<N_pn; it++){
        eta[it] = IloNumArray (env, N_pb);
        for(int pb=0; pb<N_pb; pb++){
            IloNum Aux_eta=max(0.0,normal_dist(gen, mu_P, sigma_P));
            double eta_raw = std::round(Aux_eta);
            if (eta_raw <= 47) {
                eta[it][pb] = 20;
            } else if (eta_raw == 48) {
                eta[it][pb] = 30;
            } else if (eta_raw == 49) {
                eta[it][pb] = 40;
            } else if (eta_raw == 50) {
                eta[it][pb] = 50;
            } else if (eta_raw == 51) {
                eta[it][pb] = 60;
            } else if (eta_raw == 52) {
                eta[it][pb] = 70;
            } else {
                eta[it][pb] = 80;
            }
        }
    }
    for (int pb = 0; pb < N_pn; pb++) {
        cout << "eta[" << pb << "] = " << eta[pb] << endl;   
    }
    
    BuildKappaTauMappings(1e6);

    Expdem = 0;
    ExpDemItem = IloNumArray(env, N_it);

    // double mu_D = 100;       // Mean of Demand
    sigma_D = 50;    // Std deviation of Demand
    // double rho = -0.7;        // Correlation coefficient

    d_xi = IloNumArray3 (env, N_tp+1);
    for (int t = 0; t <= N_tp; t++) {

        f[t] = IloNumArray (env, N_it);
        p[t] = IloNumArray (env, N_it);
        h[t] = IloNumArray (env, N_it);
        b[t] = IloNumArray (env, N_it);

        for (int it = 0; it < N_it; it++) {
            f[t][it] = 2000;
            p[t][it] = 10;
            h[t][it] = 5;
            b[t][it] = lost;
        }

        d[t] = IloNumArray3 (env, N_sc);
        d_xi[t] = IloNumArray2 (env, N_sc);
        for (int sc = 0; sc < N_sc; sc++) {
            d[t][sc] = IloNumArray2 (env, N_pb);
            d_xi[t][sc] = IloNumArray (env, N_it);
            for (int pb = 0; pb < N_pb; ++pb) {
                d[t][sc][pb] = IloNumArray (env, N_it);
            }
        }
    }

    W_it_pn.assign(N_it, std::vector<double>(N_pn, 0.0));
    RhoItPn.assign(N_it, std::vector<double>(N_pn, 0.0));
    std::uniform_real_distribution<double> wDist(0.5, 1.0);
    std::uniform_real_distribution<double> rhoDist(-0.7, 0.7);
    if (N_it > N_pn && N_pn > 0) {
        for (int it = 0; it < N_it; ++it) {
            int pn = it % N_pn;
            W_it_pn[it][pn] = wDist(gen);
            RhoItPn[it][pn] = rhoDist(gen);
        }
    } else if (N_it < N_pn && N_pn > 0) {
        int Kmax = std::min(6, N_pn);
        int Kmin = std::min(3, Kmax);
        std::vector<int> idx(N_pn);
        for (int i = 0; i < N_pn; ++i) idx[i] = i;
        for (int it = 0; it < N_it; ++it) {
            std::shuffle(idx.begin(), idx.end(), gen);
            std::uniform_int_distribution<int> kDist(Kmin, Kmax);
            int K = kDist(gen);
            double sumw2 = 0.0;
            for (int k = 0; k < K; ++k) {
                int pn = idx[k];
                double w = wDist(gen);
                W_it_pn[it][pn] = w;
                RhoItPn[it][pn] = rhoDist(gen);
                sumw2 += w * w;
            }
            if (sumw2 > 1.0) {
                double inv = 1.0 / std::sqrt(sumw2);
                for (int pn = 0; pn < N_pn; ++pn) {
                    W_it_pn[it][pn] *= inv;
                }
            }
        }
    } else if (N_it == N_pn && N_pn > 0) {
        if (CorrMode == 0) {
            for (int it = 0; it < N_it; ++it) {
                int pn = it % N_pn;
                W_it_pn[it][pn] = wDist(gen);
                RhoItPn[it][pn] = rhoDist(gen);
            }
        } else {
            int Kmax = std::min(6, N_pn);
            int Kmin = std::min(3, Kmax);
            std::vector<int> idx(N_pn);
            for (int i = 0; i < N_pn; ++i) idx[i] = i;
            for (int it = 0; it < N_it; ++it) {
                std::shuffle(idx.begin(), idx.end(), gen);
                std::uniform_int_distribution<int> kDist(Kmin, Kmax);
                int K = kDist(gen);
                double sumw2 = 0.0;
                for (int k = 0; k < K; ++k) {
                    int pn = idx[k];
                    double w = wDist(gen);
                    W_it_pn[it][pn] = w;
                    RhoItPn[it][pn] = rhoDist(gen);
                    sumw2 += w * w;
                }
                if (sumw2 > 1.0) {
                    double inv = 1.0 / std::sqrt(sumw2);
                    for (int pn = 0; pn < N_pn; ++pn) {
                        W_it_pn[it][pn] *= inv;
                    }
                }
            }
        }
    }
    cout << "W_it_pn (items x components):" << endl;
    for (int it = 0; it < N_it; ++it) {
        cout << "it " << it << ": ";
        for (int pn = 0; pn < N_pn; ++pn) {
            cout << W_it_pn[it][pn];
            if (pn < N_pn - 1) cout << " ";
        }
        cout << endl;
    }
    cout << "Rho_it_pn (items x components):" << endl;
    for (int it = 0; it < N_it; ++it) {
        cout << "it " << it << ": ";
        for (int pn = 0; pn < N_pn; ++pn) {
            double rho_val = (W_it_pn[it][pn] == 0.0) ? 0.0 : RhoItPn[it][pn];
            RhoItPn[it][pn] = rho_val;
            cout << rho_val;
            if (pn < N_pn - 1) cout << " ";
        }
        cout << endl;
    }

    for (int t = 1; t <= N_tp; t++) {
        for (int sc = 0; sc < N_sc; sc++) {
            for (int it = 0; it < N_it; it++) {
                d_xi[t][sc][it] = max(0.0, normal_dist(gen, mu_D, sigma_D));
            }
        }
        for (int pb = 0; pb < N_pb; ++pb) {
            for (int it = 0; it < N_it; it++) {
                double mean_shift = 0.0;
                double var_factor = 1.0;
                for (int pn = 0; pn < N_pn; ++pn) {
                    double w = W_it_pn[it][pn];
                    if (w == 0.0) continue;
                    double rho_itpn = RhoItPn[it][pn];
                    rho_itpn = std::max(-0.999, std::min(0.999, rho_itpn));
                    mean_shift += w * rho_itpn * (sigma_D / sigma_P) * (eta[pn][pb] - mu_P);
                    var_factor -= w * w * rho_itpn * rho_itpn;
                }
                if (var_factor < 1e-6) var_factor = 1e-6;
                double conditional_mean = mu_D + mean_shift;
                double conditional_std = sigma_D * std::sqrt(var_factor);
                for (int sc = 0; sc < N_sc; sc++) {
                    d[t][sc][pb][it] = max(0.0,normal_dist(gen, conditional_mean, conditional_std));
                    Expdem+=d[t][sc][pb][it];
                    ExpDemItem[it]+=d[t][sc][pb][it];
                }
            }
        }
    }
}

// Generate conditional probabilities Prob_xi_given_eta.
void Probability_generation(IloEnv env, double mu_P, double sigma_P, double sigma_D){
    Prob_xi_given_eta = IloNumArray2(env, N_pb);
    std::vector<std::vector<double>> xi_avg(N_sc, std::vector<double>(N_it, mu_D));
    for (int sc = 0; sc < N_sc; ++sc) {
        for (int it = 0; it < N_it; ++it) {
            double sum = 0.0;
            for (int t = 1; t <= N_tp; ++t) {
                sum += d_xi[t][sc][it];
            }
            xi_avg[sc][it] = sum / static_cast<double>(std::max(1, N_tp));
        }
    }

    for (int pb = 0; pb < N_pb; ++pb) {
        Prob_xi_given_eta[pb] = IloNumArray(env, N_sc);
        double sumProb = 0.0;
        for (int sc = 0; sc < N_sc; ++sc) {
            double logw = 0.0;
            for (int it = 0; it < N_it; ++it) {
                double mean_shift = 0.0;
                double var_factor = 1.0;
                for (int pn = 0; pn < N_pn; ++pn) {
                    double w = W_it_pn[it][pn];
                    if (w == 0.0) continue;
                    double rho_itpn = RhoItPn[it][pn];
                    rho_itpn = std::max(-0.999, std::min(0.999, rho_itpn));
                    mean_shift += w * rho_itpn * (sigma_D / sigma_P) * (eta[pn][pb] - mu_P);
                    var_factor -= w * w * rho_itpn * rho_itpn;
                }
                if (var_factor < 1e-6) var_factor = 1e-6;
                double mean_cond = mu_D + mean_shift;
                double var_cond = sigma_D * sigma_D * var_factor;
                double diff = xi_avg[sc][it] - mean_cond;
                logw += -0.5 * (diff * diff) / std::max(1e-8, var_cond);
            }
            double prob = std::exp(logw);
            Prob_xi_given_eta[pb][sc] = prob;
            sumProb += prob;
        }
        if (sumProb > 0.0) {
            for (int sc = 0; sc < N_sc; ++sc) {
                Prob_xi_given_eta[pb][sc] /= sumProb;
            }
        }
    }
    // cout << Prob_xi_given_eta << endl;
}

// Generate capacity and Big-M values.
void BigM_Cap_generation(IloEnv env, double beta, IloNum Expdem){
    Cap = beta*Expdem/(N_tp*N_sc*N_pb);
    // BigM=Cap;
    cout << "capacity " << Cap <<endl;

    BigM = IloNumArray3 (env, N_it);
    for (int it = 0; it < N_it; ++it) {
        BigM[it] = IloNumArray2 (env, N_pb);
        for (int pb = 0; pb < N_pb; pb++) {
            BigM[it][pb] = IloNumArray (env, N_tp+1);
        }
    }

    BigM_np = IloNumArray2 (env, N_it);
    for (int it = 0; it < N_it; ++it) {
        BigM_np[it] = IloNumArray (env, N_tp+1);
    }


    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; t++) {
            BigM_np[it][t]=0;
            for (int pb = 0; pb < N_pb; pb++) {
                BigM[it][pb][t] =0;
                for(int sc=0; sc<N_sc; sc++){
                    IloNum Aux_sc = 0;
                    for(int k=t; k<=N_tp; k++){
                        Aux_sc +=d[k][sc][pb][it] ;
                    }
                    if(Aux_sc>BigM[it][pb][t]){
                        BigM[it][pb][t] = Aux_sc;
                    }
                }
                if(BigM_np[it][t]<BigM[it][pb][t]){
                    BigM_np[it][t]=BigM[it][pb][t];
                }
            }
        }
    }
}

static double Twostage_solution_item(IloEnv env, int item){
    const IloNum rho_pb = static_cast<IloNum>(1) / static_cast<IloNum>(N_pb);

    IloNumVarArray X(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp+1, 0, 1, ILOBOOL);
    IloNumVarArray2 I(env, N_pb);
    IloNumVarArray2 L(env, N_pb);
    for (int pb = 0; pb < N_pb; ++pb) {
        I[pb] = IloNumVarArray(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
        L[pb] = IloNumVarArray(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int pb = 0; pb < N_pb; ++pb) {
        for (int t = 1; t <= N_tp; ++t) {
            obj += rho_pb * (h[t][item] * I[pb][t] + b[t][item] * L[pb][t]);
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb = 0; pb < N_pb; ++pb) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= std::min(Cap, minBigM) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int pb = 0; pb < N_pb; ++pb) {
        std::vector<IloNum> dbar(N_tp+1, 0.0);
        for (int t = 1; t <= N_tp; ++t) {
            IloNum sum = 0.0;
            for (int sc = 0; sc < N_sc; ++sc) {
                sum += d[t][sc][pb][item];
            }
            dbar[t] = (N_sc > 0) ? (sum / static_cast<IloNum>(N_sc)) : 0.0;
        }
        for (int t = 1; t <= N_tp; ++t) {
            model.add(I[pb][t] == I[pb][t-1] + X[t] - dbar[t] + L[pb][t]);
        }
        model.add(I[pb][0] == 0);
        model.add(L[pb][0] == 0);
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 1);
    cplex.setParam(IloCplex::TiLim, 60);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    double val = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return val;
}

static double Subproblem_item(IloEnv env, int item, int pb){
    const IloNum rho_pb = static_cast<IloNum>(1) / static_cast<IloNum>(N_pb);

    IloNumVarArray X(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp+1, 0, 1, ILOBOOL);
    IloNumVarArray I(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray L(env, N_tp+1, 0, IloInfinity, ILOFLOAT);

    IloModel model(env);
    IloCplex cplex(model);

    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += rho_pb * (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int t = 1; t <= N_tp; ++t) {
        obj += rho_pb * (h[t][item] * I[t] + b[t][item] * L[t]);
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        model.add(X[t] <= std::min(Cap, (int)BigM[item][pb][t]) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    std::vector<IloNum> dbar(N_tp+1, 0.0);
    for (int t = 1; t <= N_tp; ++t) {
        IloNum sum = 0.0;
        for (int sc = 0; sc < N_sc; ++sc) {
            sum += d[t][sc][pb][item];
        }
        dbar[t] = (N_sc > 0) ? (sum / static_cast<IloNum>(N_sc)) : 0.0;
    }
    for (int t = 1; t <= N_tp; ++t) {
        model.add(I[t] == I[t-1] + X[t] - dbar[t] + L[t]);
    }
    model.add(I[0] == 0);
    model.add(L[0] == 0);

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 1);
    cplex.setParam(IloCplex::TiLim, 60);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    double val = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return val;
}

// Generate probing costs (alpha) and Lambda.
void ProbingCost_generation(IloEnv env, const IloNumArray& ExpDemItem, IloNum Expdem, double sigma_D){
    IloNumArray UpperB_item (env, N_it);
    IloNumArray LowerB_item (env, N_it);
    IloNumArray2 Relax_item (env, N_it);

    IloNum Best_item=0;
    IloNumArray Diff (env, N_it);

    for (int it = 0; it < N_it; ++it) {
        Best_item = 100000;
        Relax_item[it] = IloNumArray (env, N_pb);
        UpperB_item[it] = Twostage_solution_item(env, it);
        LowerB_item[it] = 0;

        for (int pb = 0; pb < N_pb; pb++) {
            Relax_item[it][pb] = Subproblem_item (env, it, pb);
            LowerB_item[it]+= Relax_item[it][pb];
        }

        Diff[it]=(UpperB_item[it]-LowerB_item[it]);
    }
    cout << "Upper bounds: " << UpperB_item << endl;
    cout << "Lower bounds: " << LowerB_item << endl;
    cout << Diff << endl;

    alpha = IloNumArray (env, N_pn);
    IloNum sumA = 0;
    IloNum diffAvg = 0.0;
    IloNum upperAvg = 0.0;
    for (int it = 0; it < N_it; ++it) {
        diffAvg += Diff[it];
        upperAvg += UpperB_item[it];
    }
    diffAvg = (N_it > 0) ? (diffAvg / static_cast<IloNum>(N_it)) : 0.0;
    upperAvg = (N_it > 0) ? (upperAvg / static_cast<IloNum>(N_it)) : 0.0;
    if (diffAvg <= 1e-9) {
        IloNum fallback = (upperAvg > 0.0) ? (0.05 * upperAvg) : 1.0;
        diffAvg = 0.01*fallback;
        cout << "ProbingCost_generation: Diff avg ~0, using fallback=" << diffAvg << endl;
    }
    auto randu_double = [](double a, double b) -> double {
        if (b < a) {
            double tmp = a;
            a = b;
            b = tmp;
        }
        if (a == b) return a;
        double u = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        return a + (b - a) * u;
    };
    for (int j = 0; j < N_pn; ++j) {
        IloNum diffVal = (j < N_it) ? Diff[j] : diffAvg;
        if (diffVal < 0.0) diffVal = 0.0;
        if (diffVal <= 1e-9) diffVal = diffAvg;
        double lo = (1 - al) * diffVal;
        double hi = (1 + al) * diffVal;
        alpha[j] = alphaScale * randu_double(lo, hi);
        sumA+=alpha[j];
    }
    // cout << alpha << endl;

    cout << "································######################################333" << endl;
    cout << "Probing Costs alpha: " << alpha << endl;

    UpperB_item.end();
    LowerB_item.end();
    Relax_item.end();
}

// Generate all scenario data (eta, xi, costs, probabilities, and Big-Ms).
void Instance_Generation (IloEnv env, std::mt19937& gen, double beta){
    IloNum Expdem = 0.0;
    IloNumArray ExpDemItem(env);
    double mu_P = 0.0;
    double sigma_P = 0.0;
    double sigma_D = 0.0;

    scenarios_generation(env, gen, Expdem, ExpDemItem, mu_P, sigma_P, sigma_D);
    Probability_generation(env, mu_P, sigma_P, sigma_D);
    BigM_Cap_generation(env, beta, Expdem);
    ProbingCost_generation(env, ExpDemItem, Expdem, sigma_D);
}

struct EtaKey {
    std::vector<long long> v;
    bool operator==(EtaKey const& other) const { return v == other.v; }
};

struct EtaKeyHash {
    std::size_t operator()(EtaKey const& k) const noexcept {
        // Simple hash combine
        std::size_t h = 0;
        for (auto x : k.v) {
            h ^= std::hash<long long>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

static inline int zBit(int zMask, int it) {
    return (zMask >> it) & 1;
}

// Build tau class mappings for each probing vector (zMask).
void BuildKappaTauMappings(double roundingScale) {
    
    // Number of probing vectors z in {0,1}^{N_it}
    N_z = 1 << N_pn;
    EtaRoundingScale = roundingScale;

    KappaTau.assign(N_z, std::vector<int>(N_pb, -1));
    TauClassId.assign(N_z, std::vector<int>(N_pb, -1));
    TauReps.assign(N_z, std::vector<int>()); // variable length per z

    for (int zMask = 0; zMask < N_z; ++zMask) {

        // Map from "observed signature" -> classId
        std::unordered_map<EtaKey, int, EtaKeyHash> keyToClass;
        std::vector<int> reps; // reps[classId] = representative gamma (scenario index)

        int nextClass = 0;

        for (int gamma = 0; gamma < N_pb; ++gamma) {

            EtaKey key;
            key.v.reserve(N_pn);

            // Signature = rounded eta[it][gamma] for all probed items (z_i=1)
            for (int it = 0; it < N_pn; ++it) {
                if (zBit(zMask, it)) {
                    double val = eta[it][gamma];
                    long long q = llround(val * roundingScale);
                    key.v.push_back(q);
                }
            }

            auto itFound = keyToClass.find(key);
            int classId;

            if (itFound == keyToClass.end()) {
                classId = nextClass++;
                keyToClass.emplace(key, classId);
                reps.push_back(gamma); // choose first seen gamma as representative tau
            } else {
                classId = itFound->second;
            }

            TauClassId[zMask][gamma] = classId;
            KappaTau[zMask][gamma]   = reps[classId]; // kappa(z,gamma)=tau (representative scenario index)
        }

        TauReps[zMask] = reps;
    }
}
