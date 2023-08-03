title = "Single_Clean_PNCC_Adam_2"
cm_plot_title = "Confusion Matrix " + title
acc_plot_title = "Model Accuracy " + title

 # contoh "../dataset/pickle/mfcc/mfccs_15_IDN.pkl"
dataset = "../dataset/pickle/pncc/pncc_15_ENG.pkl"
label = "../dataset/pickle/label/label_pncc_15_ENG.pkl"

test_size = 0.2
batch_size = 32
epochs = 100
lr = 1e-3
num_folds = 3

base_path = "../saved/single/CNN/"

save_model_dir = base_path+"model/" + title

txt_output = base_path+"text/" + title + "_eval_metrics"
csv_report_output = base_path+"text/clf_report_" + title

history_output = base_path+"history/" + title

plot_model_output = base_path+"fig/model_" + title + ".png"
plot_accuracy_output = base_path+"fig/accuracy_" + title
plot_cm_output = base_path+"fig/cm_" + title

