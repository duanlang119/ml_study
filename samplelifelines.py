from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

rossi_dataset = load_rossi()


#rossi_dataset.head()
cph = CoxPHFitter()
cph.fit(rossi_dataset, duration_col='week', event_col='arrest',)

cph.print_summary()

print(cph.params_)

print(cph.baseline_survival_)