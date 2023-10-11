from mrx_link.sdk.utils import *


load_runway_dataset_code = """
import os
import pandas as pd

dfs = []
for dirname, _, filenames in os.walk(RUNWAY_DATA_PATH):
    for filename in filenames:
        dfs += [pd.read_csv(os.path.join(dirname, filename))]
df = pd.concat(dfs)
"""

set_data_code = """
proc_df = df.set_index("datetime").drop(columns=["id"]).tail(1000)
"""

split_data_code = """
from sklearn.model_selection import train_test_split

train, valid = train_test_split(proc_df, test_size=0.2)
"""

pca_class_code = """
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCADetector:
    def __init__(self, n_components):
        self._use_columns = ...
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=n_components)
    
    def fit(self, X):
        self._use_columns = X.columns
        X_scaled = self._scaler.fit_transform(X)
        self._pca.fit(X_scaled)
    
    def predict(self, X):
        X = X[self._use_columns]
        X_scaled = self._scaler.transform(X)
        recon = self._recon(X_scaled)
        recon_err = ((X_scaled - recon) ** 2).mean(1)
        recon_err_df = pd.DataFrame(recon_err, columns=["anomaly_score"], index=X.index)
        return recon_err_df
    
    def _recon(self, X):
        z = self._pca.transform(X)
        recon = self._pca.inverse_transform(z)
        return recon

    def reconstruct(self, X):
        X_scaled = self._scaler.transform(X)
        recon_scaled = self._recon(X_scaled)
        recon = self._scaler.inverse_transform(recon_scaled)
        recon_df = pd.DataFrame(recon, index=X.index, columns=X.columns)
        return recon_df
"""

train_code = """
parameters = {"n_components": N_COMPONENTS}

detector = PCADetector(n_components=parameters["n_components"])
detector.fit(train)

train_pred = detector.predict(train)
valid_pred = detector.predict(valid)

mean_train_recon_err = train_pred.mean()
mean_valid_recon_err = valid_pred.mean()
"""

input_sample_code = """
input_sample = proc_df.sample(1)
input_sample
"""

send_model_to_runway_code = """
import runway

# start run
runway.start_run()

# log model related info
runway.log_parameters(parameters)
runway.log_metric("mean_train_recon_err", mean_train_recon_err)
runway.log_metric("mean_valid_recon_err", mean_valid_recon_err)

# log model
runway.log_model(model_name="pca-model", model=detector, input_samples={"predict": input_sample})

# stop run
runway.stop_run()
"""


if __name__ == "__main__":
    # Pipeline object
    pipeline = LinkPipeline()

    # load dataset
    load_runway_dataset = create_link_component(
        identifier="load_runway_dataset", name="load_runway_dataset", code=load_runway_dataset_code
    )
    pipeline.add_component(component=load_runway_dataset)
    dataset_param = create_link_parameter(name="RUNWAY_DATA_PATH", value="./dataset")

    # set data
    set_data = create_link_component(identifier="set_data", name="set_data", code=set_data_code)
    pipeline.add_component(component=set_data)
    pipeline.add_edge(parent_id="load_runway_dataset", child_id="set_data")

    # split data
    split_data = create_link_component(identifier="split_data", name="split_data", code=split_data_code)
    pipeline.add_component(component=split_data)
    pipeline.add_edge(parent_id="set_data", child_id="split_data")

    # model class
    pca_class = create_link_component(identifier="pca_class", name="pca_class", code=pca_class_code)
    pipeline.add_component(component=pca_class)
    n_component_param = create_link_parameter(name="N_COMPONENTS", value=2)

    # train
    train = create_link_component(identifier="train", name="train", code=train_code)
    pipeline.add_component(component=train)
    pipeline.add_edge(parent_id="split_data", child_id="train")
    pipeline.add_edge(parent_id="pca_class", child_id="train")

    # input sample
    input_sample = create_link_component(identifier="input_sample", name="input_sample", code=input_sample_code)
    pipeline.add_component(component=input_sample)
    pipeline.add_edge(parent_id="train", child_id="input_sample")

    # send model to runway
    send_model_to_runway = create_link_component(
        identifier="send_model_to_runway", name="send_model_to_runway", code=send_model_to_runway_code
    )
    pipeline.add_component(component=send_model_to_runway)
    pipeline.add_edge(parent_id="input_sample", child_id="send_model_to_runway")

    # set parameters
    pipeline.set_parameters(new_parameters=[dataset_param, n_component_param])

    # print pipeline meta info
    pipeline.print()

    # run pipeline
    pipeline.execute_all()
