package Classification.Model.Svm;

import Classification.Parameter.SvmParameter;
import Util.Swap;

public class Solver {
    private int activeSize;
    private double[] y;
    private double[] alpha;
    private double[] G;
    private AlphaStatusType[] alphaStatus;
    private Q Q;
    private double C;
    private double[] b;
    private int[] activeSet;
    private double[] GBar;
    private int l;
    private boolean unshrinked = false;
    private SvmParameter parameter;

    public Solver(int l, double[] b, double[] y, SvmParameter parameter, Problem problem){
        this.parameter = parameter;
        this.l = l;
        Q = new Q(problem, parameter, y);
        this.b = b;
        this.y = y;
        C = parameter.getC();
    }

    public SolutionInfo solve(){
        double[] Qi;
        double[] Qj;
        double alphaI, oldAlphaI, oldAlphaJ, delta, diff, sum, deltaAlphaI, deltaAlphaJ;
        int counter, i, j;
        boolean upperBoundI, upperBoundJ;
        alpha = new double[l];
        for (i = 0; i < l; i++){
            alpha[i] = 0.0;
        }
        alphaStatus = new AlphaStatusType[l];
        for (i = 0; i < l; i++){
            updateAlphaStatus(i);
        }
        activeSet = new int[l];
        for (i = 0; i < l; i++){
            activeSet[i] = i;
        }
        activeSize = l;
        G = new double[l];
        GBar = new double[l];
        for (i = 0; i < l; i++){
            G[i] = b[i];
            GBar[i] = 0;
        }
        for (i = 0; i < l; i++){
            if (!isLowerBound(i)){
                Qi = getQ(i, l);
                alphaI = alpha[i];
                for (j = 0; j < l; j++){
                    G[j] += alphaI * Qi[j];
                }
                if (isUpperBound(i)){
                    for (j = 0; j < l; j++){
                        GBar[j] += C * Qi[j];
                    }
                }
            }
        }
        counter = Math.min(l, 1000) + 1;
        while (true){
            if (--counter == 0){
                counter = Math.min(l, 1000);
                if (parameter.isShrinking()){
                    doShrinking();
                }
            }
            int[] out = new int[2];
            if (selectWorkingSet(out)){
                reconstructGradient();
                activeSize = l;
                if (selectWorkingSet(out)){
                    break;
                } else {
                    counter = 1;
                }
            }
            i = out[0];
            j = out[1];
            Qi = getQ(i, activeSize);
            Qj = getQ(j, activeSize);
            oldAlphaI = alpha[i];
            oldAlphaJ = alpha[j];
            if (y[i] != y[j]){
                delta = (-G[i] - G[j]) / Math.max(Qi[i] + Qj[j] + 2 * Qi[j], 0.0);
                diff = alpha[i] - alpha[j];
                alpha[i] += delta;
                alpha[j] += delta;
                if (diff > 0){
                    if (alpha[j] < 0){
                        alpha[j] = 0;
                        alpha[i] = diff;
                    }
                    if (alpha[i] > C){
                        alpha[i] = C;
                        alpha[j] = C - diff;
                    }
                } else {
                    if (alpha[i] < 0){
                        alpha[i] = 0;
                        alpha[j] = -diff;
                    }
                    if (alpha[j] > C){
                        alpha[j] = C;
                        alpha[i] = C + diff;
                    }
                }
            } else {
                delta = (G[i] - G[j]) / Math.max(Qi[i] + Qj[j] - 2 * Qi[j], 0.0);
                sum = alpha[i] + alpha[j];
                alpha[i] -= delta;
                alpha[j] += delta;
                if (sum > C){
                    if (alpha[i] > C){
                        alpha[i] = C;
                        alpha[j] = sum - C;
                    }
                    if (alpha[j] > C){
                        alpha[j] = C;
                        alpha[i] = sum - C;
                    }
                } else {
                    if (alpha[j] < 0){
                        alpha[j] = 0;
                        alpha[i] = sum;
                    }
                    if (alpha[i] < 0){
                        alpha[i] = 0;
                        alpha[j] = sum;
                    }
                }
            }
            deltaAlphaI = alpha[i] - oldAlphaI;
            deltaAlphaJ = alpha[j] - oldAlphaJ;
            for (int k = 0; k < activeSize; k++){
                G[k] += Qi[k] * deltaAlphaI + Qj[k] * deltaAlphaJ;
            }
            upperBoundI = isUpperBound(i);
            upperBoundJ = isUpperBound(j);
            updateAlphaStatus(i);
            updateAlphaStatus(j);
            if (upperBoundI != isUpperBound(i)){
                Qi = getQ(i, l);
                if (upperBoundI){
                    for (int k = 0; k < l; k++){
                        GBar[k] -= C * Qi[k];
                    }
                } else {
                    for (int k = 0; k < l; k++){
                        GBar[k] += C * Qi[k];
                    }
                }
            }
            if (upperBoundJ != isUpperBound(j)){
                Qj = getQ(j, l);
                if (upperBoundJ){
                    for (int k = 0; k < l; k++){
                        GBar[k] -= C * Qj[k];
                    }
                } else {
                    for (int k = 0; k < l; k++){
                        GBar[k] += C * Qj[k];
                    }
                }
            }
        }
        for (i = 0; i < l; i++){
            alpha[activeSet[i]] = alpha[i];
        }
        return new SolutionInfo(calculateRho(), alpha);
    }

    private void updateAlphaStatus(int i){
        if (alpha[i] >= C){
            alphaStatus[i] = AlphaStatusType.UPPER_BOUND;
        } else {
            if (alpha[i] <= 0){
                alphaStatus[i] = AlphaStatusType.LOWER_BOUND;
            } else {
                alphaStatus[i] = AlphaStatusType.FREE;
            }
        }
    }

    private boolean isUpperBound(int i){
        return alphaStatus[i].equals(AlphaStatusType.UPPER_BOUND);
    }

    private boolean isLowerBound(int i){
        return alphaStatus[i].equals(AlphaStatusType.LOWER_BOUND);
    }

    private boolean isFree(int i){
        return alphaStatus[i].equals(AlphaStatusType.FREE);
    }

    private double[] getQ(int i, int length){
        return Q.getQ(i, length);
    }

    private void swapIndex(int i, int j){
        Q.swapIndex(i, j);
        Swap.swap(y, i, j);
        Swap.swap(G, i, j);
        Swap.swap(alpha, i, j);
        Swap.swap(b, i, j);
        Swap.swap(GBar, i, j);
        Swap.swap(activeSet, i, j);
        AlphaStatusType t = alphaStatus[i];
        alphaStatus[i] = alphaStatus[j];
        alphaStatus[j] = t;
    }

    private void reconstructGradient(){
        double[] Qi;
        double alphaI;
        if (activeSize == l){
            return;
        }
        for (int i = activeSize; i < l; i++){
            G[i] = GBar[i] + b[i];
        }
        for (int i = 0; i < activeSize; i++){
            if (isFree(i)){
                Qi = getQ(i, l);
                alphaI = alpha[i];
                for (int j = activeSize; j < l; j++){
                    G[j] += alphaI * Qi[j];
                }
            }
        }
    }

    private boolean selectWorkingSet(int[] out){
        double[] GMax = {-Double.MAX_VALUE, -Double.MAX_VALUE};
        for (int i = 0; i < activeSize; i++){
            if (y[i] == 1){
                if (!isUpperBound(i)){
                    if (-G[i] >= GMax[0]){
                        GMax[0] = -G[i];
                        out[0] = i;
                    }
                }
                if (!isLowerBound(i)){
                    if (G[i] >= GMax[1]){
                        GMax[1] = G[i];
                        out[1] = i;
                    }
                }
            } else {
                if (!isUpperBound(i)){
                    if (-G[i] >= GMax[1]){
                        GMax[1] = -G[i];
                        out[1] = i;
                    }
                }
                if (!isLowerBound(i)){
                    if (G[i] >= GMax[0]){
                        GMax[0] = G[i];
                        out[0] = i;
                    }
                }
            }
        }
        if (GMax[0] + GMax[1] < 0.001){
            return true;
        }
        return false;
    }

    private double calculateRho(){
        double upperBound = Double.MAX_VALUE, lowerBound = -Double.MAX_VALUE, sumFree = 0, yG;
        int numberOfFree = 0;
        for (int i = 0; i < activeSize; i++){
            yG = y[i] * G[i];
            if (isLowerBound(i)){
                if (y[i] > 0){
                    upperBound = Math.min(upperBound, yG);
                } else {
                    lowerBound = Math.max(lowerBound, yG);
                }
            } else {
                if (isUpperBound(i)){
                    if (y[i] < 0){
                        upperBound = Math.min(upperBound, yG);
                    } else {
                        lowerBound = Math.max(lowerBound, yG);
                    }
                } else {
                    numberOfFree++;
                    sumFree += yG;
                }
            }
        }
        if (numberOfFree > 0){
            return sumFree / numberOfFree;
        } else {
            return (upperBound + lowerBound) / 2;
        }
    }

    private void doShrinking(){
        double[] Gm = new double[2];
        int[] out = new int[2];
        if (selectWorkingSet(out)){
            return;
        }
        Gm[0] = -y[out[1]] * G[out[1]];
        Gm[1] = y[out[0]] * G[out[0]];
        for (int k = 0; k < activeSize; k++){
            if (isLowerBound(k)){
                if (y[k] == 1) {
                    if (-G[k] >= Gm[0]) {
                        continue;
                    }
                } else {
                    if (-G[k] >= Gm[1]){
                        continue;
                    }
                }
            } else {
                if (isUpperBound(k)){
                    if (y[k] == 1){
                        if (G[k] >= Gm[1]){
                            continue;
                        }
                    } else {
                        if (G[k] >= Gm[0]){
                            continue;
                        }
                    }
                } else {
                    continue;
                }
            }
            activeSize--;
            swapIndex(k, activeSize);
            k--;
        }
        if (unshrinked || -(Gm[0] + Gm[1]) > 0.001 * 10){
            return;
        }
        unshrinked = true;
        reconstructGradient();
        for (int k = l - 1; k >= activeSize; k--){
            if (isLowerBound(k)){
                if (y[k] == 1) {
                    if (-G[k] < Gm[0]) {
                        continue;
                    }
                } else {
                    if (-G[k] < Gm[1]){
                        continue;
                    }
                }
            } else {
                if (isUpperBound(k)){
                    if (y[k] == 1){
                        if (G[k] < Gm[1]){
                            continue;
                        }
                    } else {
                        if (G[k] < Gm[0]){
                            continue;
                        }
                    }
                } else {
                    continue;
                }
            }
            swapIndex(k, activeSize);
            activeSize++;
            k++;
        }
    }
}
