package Classification.Model.Svm;

import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.Model.ValidatedModel;
import Classification.Parameter.Parameter;
import Classification.Parameter.SvmParameter;
import Math.DiscreteDistribution;

import java.io.Serializable;
import java.util.HashMap;

public class SvmModel extends ValidatedModel implements Serializable{
    private DiscreteDistribution classDistribution;
    private double[] rho;
    private int[] numberOfSupportVectors;
    private NodeList[] supportVectors;
    private double[][] supportVectorCoefficients;
    private SvmParameter parameter;
    private int numberOfClasses;
    private int numberOfProblems;

    @Override
    public void loadModel(String fileName) {

    }

    /**
     * Training algorithm for Support Vector Machine classifier.
     *
     * @param trainSet   Training data given to the algorithm.
     * @param parameters Parameters of the SVM classifier algorithm.
     * @throws DiscreteFeaturesNotAllowed Exception for discrete features.
     */
    public void train(InstanceList trainSet, Parameter parameters) throws DiscreteFeaturesNotAllowed {
        int[] start;
        int[] nonZeroCount;
        int[] nonZeroStart;
        int l, nSV, si, ci, sj, cj, p, q, totalSupportVectors;
        NodeList[] x;
        NodeList[] subProblemX;
        double[] subProblemY;
        boolean[] nonZero;
        SolutionInfo[] weights;
        if (!discreteCheck(trainSet.get(0))) {
            throw new DiscreteFeaturesNotAllowed();
        }
        this.parameter = (SvmParameter) parameters;
        trainSet.sort();
        Problem problem = new Problem(trainSet);
        l = problem.getL();
        classDistribution = trainSet.classDistribution();
        numberOfClasses = classDistribution.size();
        start = groupClasses();
        x = new NodeList[l];
        nonZero = new boolean[l];
        for (int i = 0; i < l; i++){
            x[i] = problem.getX()[i];
            nonZero[i] = false;
        }
        p = 0;
        numberOfProblems = (numberOfClasses * (numberOfClasses - 1)) / 2;
        weights = new SolutionInfo[numberOfProblems];
        for (int i = 0; i < numberOfClasses; i++){
            for (int j = i + 1; j < numberOfClasses; j++){
                si = start[i];
                sj = start[j];
                ci = classDistribution.getValue(i);
                cj = classDistribution.getValue(j);
                subProblemX = new NodeList[ci + cj];
                subProblemY = new double[ci + cj];
                for (int k = 0; k < ci; k++){
                    subProblemX[k] = x[si + k];
                    subProblemY[k] = 1;
                }
                for (int k = 0; k < cj; k++){
                    subProblemX[ci + k] = x[sj + k];
                    subProblemY[ci + k] = -1;
                }
                weights[p] = solveSingle(new Problem(subProblemX, subProblemY));
                for (int k = 0; k < ci; k++){
                    if (!nonZero[si + k] && Math.abs(weights[p].getAlpha(k)) > 0){
                        nonZero[si + k] = true;
                    }
                }
                for (int k = 0; k < cj; k++){
                    if (!nonZero[sj + k] && Math.abs(weights[p].getAlpha(ci + k)) > 0){
                        nonZero[sj + k] = true;
                    }
                }
                p++;
            }
        }
        rho = new double[numberOfProblems];
        for (int i = 0; i < numberOfProblems; i++){
            rho[i] = weights[i].getRho();
        }
        totalSupportVectors = 0;
        nonZeroCount = new int[numberOfClasses];
        numberOfSupportVectors = new int[numberOfClasses];
        for (int i = 0; i < numberOfClasses; i++){
            nSV = 0;
            for (int j = 0; j < classDistribution.getValue(i); j++){
                if (nonZero[start[i] + j]){
                    nSV++;
                    totalSupportVectors++;
                }
            }
            numberOfSupportVectors[i] = nSV;
            nonZeroCount[i] = nSV;
        }
        supportVectors = new NodeList[totalSupportVectors];
        p = 0;
        for (int i = 0; i < l; i++){
            if (nonZero[i]){
                supportVectors[p] = x[i];
                p++;
            }
        }
        nonZeroStart = new int[numberOfClasses];
        nonZeroStart[0] = 0;
        for (int i = 1; i < numberOfClasses; i++){
            nonZeroStart[i] = nonZeroStart[i - 1] + nonZeroCount[i - 1];
        }
        supportVectorCoefficients = new double[numberOfClasses - 1][];
        for (int i = 0; i < numberOfClasses - 1; i++){
            supportVectorCoefficients[i] = new double[totalSupportVectors];
        }
        p = 0;
        for (int i = 0; i < numberOfClasses; i++){
            for (int j = i + 1; j < numberOfClasses; j++){
                si = start[i];
                sj = start[j];
                ci = classDistribution.getValue(i);
                cj = classDistribution.getValue(j);
                q = nonZeroStart[i];
                for (int k = 0; k < ci; k++){
                    if (nonZero[si + k]){
                        supportVectorCoefficients[j - 1][q] = weights[p].getAlpha(k);
                        q++;
                    }
                }
                q = nonZeroStart[j];
                for (int k = 0; k < cj; k++){
                    if (nonZero[sj + k]){
                        supportVectorCoefficients[i][q] = weights[p].getAlpha(ci + k);
                        q++;
                    }
                }
                p++;
            }
        }
    }

    private int[] groupClasses(){
        int[] start;
        start = new int[numberOfClasses];
        start[0] = 0;
        for (int i = 1; i < numberOfClasses; i++){
            start[i] = start[i - 1] + classDistribution.getValue(i - 1);
        }
        return start;
    }

    private SolutionInfo solveSingle(Problem problem){
        double[] minusOnes;
        double[] y;
        Solver solver;
        SolutionInfo solutionInfo;
        minusOnes = new double[problem.getL()];
        y = new double[problem.getL()];
        for (int i = 0; i < problem.getL(); i++){
            minusOnes[i] = -1;
            if (problem.getY()[i] > 0){
                y[i] = 1;
            } else {
                y[i] = -1;
            }
        }
        solver = new Solver(problem.getL(), minusOnes, y, parameter, problem);
        solutionInfo = solver.solve();
        for (int i = 0; i < problem.getL(); i++){
            solutionInfo.setAlpha(i, solutionInfo.getAlpha(i) * y[i]);
        }
        return solutionInfo;
    }

    private double[] predictValues(NodeList x){
        int[] start;
        int p, si, sj, ci, cj;
        int l = supportVectors.length;
        double[] coefficients1;
        double[] coefficients2;
        double[] kernelValues = new double[l];
        double[] result;
        for (int i = 0; i < l; i++){
            kernelValues[i] = Kernel.function(x, supportVectors[i], parameter);
        }
        start = new int[numberOfClasses];
        start[0] = 0;
        for (int i = 1; i < numberOfClasses; i++){
            start[i] = start[i - 1] + numberOfSupportVectors[i - 1];
        }
        result = new double[numberOfProblems];
        p = 0;
        for (int i = 0; i < numberOfClasses; i++){
            for (int j = i + 1; j < numberOfClasses; j++){
                double sum = 0;
                si = start[i];
                sj = start[j];
                ci = numberOfSupportVectors[i];
                cj = numberOfSupportVectors[j];
                coefficients1 = supportVectorCoefficients[j - 1];
                coefficients2 = supportVectorCoefficients[i];
                for (int k = 0; k < ci; k++){
                    sum += coefficients1[si + k] * kernelValues[si + k];
                }
                for (int k = 0; k < cj; k++){
                    sum += coefficients2[sj + k] * kernelValues[sj + k];
                }
                sum -= rho[p];
                result[p] = sum;
                p++;
            }
        }
        return result;
    }

    public String predict(Instance instance) {
        int pos, maxIndex, maxVotes;
        int[] numberOfVotes = new int[numberOfClasses];
        NodeList x = instance.toNodeList();
        double[] predictedValues = predictValues(x);
        for (int i = 0; i < numberOfClasses; i++){
            numberOfVotes[i] = 0;
        }
        pos = 0;
        for (int i = 0; i < numberOfClasses; i++){
            for (int j = i + 1; j < numberOfClasses; j++){
                if (predictedValues[pos] > 0){
                    numberOfVotes[i]++;
                } else {
                    numberOfVotes[j]++;
                }
                pos++;
            }
        }
        maxVotes = numberOfVotes[0];
        maxIndex = 0;
        for (int i = 1; i < numberOfClasses; i++){
            if (numberOfVotes[i] > maxVotes){
                maxIndex = i;
                maxVotes = numberOfVotes[i];
            }
        }
        return classDistribution.getItem(maxIndex);
    }

    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        return null;
    }

    @Override
    public void saveTxt(String fileName) {

    }

}
