package Classification.Model.Ensemble;

import Classification.Model.Model;

import java.util.ArrayList;

public abstract class EnsembleModel extends Model {

    protected final ArrayList<Model> models;

    public EnsembleModel(ArrayList<Model> models) {
        this.models = models;
    }

    @Override
    public void saveTxt(String fileName) {

    }
}
