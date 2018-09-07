package Classification.StatisticalTest;

import Classification.Attribute.AttributeType;
import Classification.Classifier.DiscreteFeaturesNotAllowed;
import Classification.Classifier.Dummy;
import Classification.Classifier.NaiveBayes;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.Experiment.Experiment;
import Classification.Experiment.StratifiedKFoldRun;
import Classification.Performance.ExperimentPerformance;
import Classification.Parameter.Parameter;

import java.util.ArrayList;

public class TestStatisticalTest {

    private static Parameter defaultParameter(int seed) { return new Parameter(seed);}

    private static DataSet readIris(){
        ArrayList<AttributeType> attributeTypes = new ArrayList<>();
        for (int i = 0; i < 4; i++){
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        DataDefinition dataDefinition = new DataDefinition(attributeTypes);
        return new DataSet(dataDefinition, ",", "Data/Classification/iris.data");
    }

    public static void main(String[] args){
        DataSet dataSet = readIris();
        try {
            Parameter parameter1 = defaultParameter(1);
            StratifiedKFoldRun run = new StratifiedKFoldRun(10);
            ExperimentPerformance results1 = run.execute(new Experiment(new NaiveBayes(), parameter1, dataSet));
            System.out.println(results1.meanClassificationPerformance().getErrorRate());
            Parameter parameter2 = defaultParameter(14);
            ExperimentPerformance results2 = run.execute(new Experiment(new NaiveBayes(), parameter2, dataSet));
            System.out.println(results2.meanClassificationPerformance().getErrorRate());
            Parameter parameter3 = defaultParameter(1);
            ExperimentPerformance results3 = run.execute(new Experiment(new Dummy(), parameter3, dataSet));
            System.out.println(results3.meanClassificationPerformance().getErrorRate());
            StatisticalTestResult result = new Paired5x2t().compare(results1, results2);
            System.out.println(result.getPValue());
            System.out.println(result.oneTailed(0.05));
            System.out.println(result.twoTailed(0.05));
            StatisticalTestResult result2 = new Paired5x2t().compare(results3, results1);
            System.out.println(result2.getPValue());
            System.out.println(result2.oneTailed(0.05));
            System.out.println(result2.twoTailed(0.05));
        } catch (StatisticalTestNotApplicable | DiscreteFeaturesNotAllowed statisticalTestNotApplicable) {
            statisticalTestNotApplicable.printStackTrace();
        }
    }
}
