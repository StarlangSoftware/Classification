package Classification.Performance;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class ExperimentPerformance implements Comparable {
    private final ArrayList<Performance> results;
    private boolean containsDetails = true;
    private boolean classification = true;

    /**
     * A constructor which creates a new {@link ArrayList} of {@link Performance} as results.
     */
    public ExperimentPerformance() {
        results = new ArrayList<>();
    }

    /**
     * A constructor that takes a file name as an input and takes the inputs from that file assigns these inputs to the errorRate
     * and adds them to the results {@link ArrayList} as a new {@link Performance}.
     *
     * @param fileName String input.
     * @throws FileNotFoundException if a pathname fails.
     */
    public ExperimentPerformance(String fileName) throws FileNotFoundException {
        results = new ArrayList<>();
        containsDetails = false;
        Scanner input = new Scanner(new File(fileName));
        while (input.hasNext()) {
            String performance = input.nextLine();
            results.add(new Performance(Double.parseDouble(performance)));
        }
    }

    /**
     * The add method takes a {@link Performance} as an input and adds it to the results {@link ArrayList}.
     *
     * @param performance {@link Performance} input.
     */
    public void add(Performance performance) {
        if (!(performance instanceof DetailedClassificationPerformance)) {
            containsDetails = false;
        }
        if (!(performance instanceof ClassificationPerformance)) {
            classification = false;
        }
        results.add(performance);
    }

    /**
     * The numberOfExperiments method returns the size of the results {@link ArrayList}.
     *
     * @return The results {@link ArrayList}.
     */
    public int numberOfExperiments() {
        return results.size();
    }

    /**
     * The getErrorRate method takes an index as an input and returns the errorRate at given index of results {@link ArrayList}.
     *
     * @param index Index of results {@link ArrayList} to retrieve.
     * @return The errorRate at given index of results {@link ArrayList}.
     */
    public double getErrorRate(int index) {
        return results.get(index).getErrorRate();
    }

    /**
     * The getAccuracy method takes an index as an input. It returns the accuracy of a {@link Performance} at given index of results {@link ArrayList}.
     *
     * @param index Index of results {@link ArrayList} to retrieve.
     * @return The accuracy of a {@link Performance} at given index of results {@link ArrayList}.
     * @throws ClassificationAlgorithmExpectedException returns "Classification Algorithm required for accuracy metric" String.
     */
    public double getAccuracy(int index) throws ClassificationAlgorithmExpectedException {
        if (results.get(index) instanceof ClassificationPerformance) {
            return ((ClassificationPerformance) results.get(index)).getAccuracy();
        } else {
            throw new ClassificationAlgorithmExpectedException();
        }
    }

    /**
     * The meanPerformance method loops through the performances of results {@link ArrayList} and sums up the errorRates of each then
     * returns a new {@link Performance} with the mean of that summation.
     *
     * @return A new {@link Performance} with the mean of the summation of errorRates.
     */
    public Performance meanPerformance() {
        double sumError = 0;
        for (Performance performance : results) {
            sumError += performance.getErrorRate();
        }
        return new Performance(sumError / results.size());
    }

    /**
     * The meanClassificationPerformance method loops through the performances of results {@link ArrayList} and sums up the accuracy of each
     * classification performance, then returns a new classificationPerformance with the mean of that summation.
     *
     * @return A new classificationPerformance with the mean of that summation.
     */
    public ClassificationPerformance meanClassificationPerformance() {
        if (results.isEmpty() || !classification) {
            return null;
        }
        double sumAccuracy = 0;
        for (Performance performance : results) {
            ClassificationPerformance classificationPerformance = (ClassificationPerformance) performance;
            sumAccuracy += classificationPerformance.getAccuracy();
        }
        return new ClassificationPerformance(sumAccuracy / results.size());
    }

    /**
     * The meanDetailedPerformance method gets the first confusion matrix of results {@link ArrayList}.
     * Then, it adds new confusion matrices as the {@link DetailedClassificationPerformance} of
     * other elements of results ArrayList' confusion matrices as a {@link DetailedClassificationPerformance}.
     *
     * @return A new {@link DetailedClassificationPerformance} with the {@link ConfusionMatrix} sum.
     */
    public DetailedClassificationPerformance meanDetailedPerformance() {
        if (results.isEmpty() || !containsDetails) {
            return null;
        }
        ConfusionMatrix sum = ((DetailedClassificationPerformance) results.get(0)).getConfusionMatrix();
        for (int i = 1; i < results.size(); i++) {
            sum.addConfusionMatrix(((DetailedClassificationPerformance) results.get(i)).getConfusionMatrix());
        }
        return new DetailedClassificationPerformance(sum);
    }

    /**
     * The standardDeviationPerformance method loops through the {@link Performance}s of results {@link ArrayList} and returns
     * a new Performance with the standard deviation.
     *
     * @return A new Performance with the standard deviation.
     */
    public Performance standardDeviationPerformance() {
        double sumErrorRate = 0;
        Performance averagePerformance;
        averagePerformance = meanPerformance();
        for (Performance performance : results) {
            sumErrorRate += Math.pow(performance.getErrorRate() - averagePerformance.getErrorRate(), 2);
        }
        return new Performance(Math.sqrt(sumErrorRate / (results.size() - 1)));
    }

    /**
     * The standardDeviationClassificationPerformance method loops through the {@link Performance}s of results {@link ArrayList} and
     * returns a new {@link ClassificationPerformance} with standard deviation.
     *
     * @return A new {@link ClassificationPerformance} with standard deviation.
     */
    public ClassificationPerformance standardDeviationClassificationPerformance() {
        if (results.isEmpty() || !classification) {
            return null;
        }
        double sumAccuracy = 0, sumErrorRate = 0;
        ClassificationPerformance averageClassificationPerformance;
        averageClassificationPerformance = meanClassificationPerformance();
        for (Performance performance : results) {
            ClassificationPerformance classificationPerformance = (ClassificationPerformance) performance;
            sumAccuracy += Math.pow(classificationPerformance.getAccuracy() - averageClassificationPerformance.getAccuracy(), 2);
            sumErrorRate += Math.pow(classificationPerformance.getErrorRate() - averageClassificationPerformance.getErrorRate(), 2);
        }
        return new ClassificationPerformance(Math.sqrt(sumAccuracy / (results.size() - 1)), Math.sqrt(sumErrorRate / (results.size() - 1)));
    }

    /**
     * The isBetter method  takes an {@link ExperimentPerformance} as an input and returns true if the result of compareTo method is positive
     * and false otherwise.
     *
     * @param experimentPerformance {@link ExperimentPerformance} input.
     * @return True if the result of compareTo method is positive and false otherwise.
     */
    public boolean isBetter(ExperimentPerformance experimentPerformance) {
        return compareTo(experimentPerformance) > 0;
    }

    /**
     * The compareTo method takes an {@link Object} as an input and compares the accuracy of given object and results {@link ArrayList}.
     *
     * @param o {@link Object} to compare with.
     * @return Returns a positive value if given object;s accuracy is less than the results {@link ArrayList}'s accuracy, false otherwise, and 0 if
     * they are equal.
     */
    @Override
    public int compareTo(Object o) {
        if (o instanceof ExperimentPerformance) {
            double accuracy1, accuracy2;
            accuracy1 = meanClassificationPerformance().getAccuracy();
            accuracy2 = ((ExperimentPerformance) o).meanClassificationPerformance().getAccuracy();
            return Double.compare(accuracy1, accuracy2);
        } else {
            return 0;
        }
    }
}
