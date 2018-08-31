import xlsxwriter
from collections import namedtuple
import importlib

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('Results.xlsx')
worksheets = [workbook.add_worksheet(), workbook.add_worksheet()]

# Name and colour of the corresponding row
experiments = [
    ('model_1', None),
    ('model_2', None),
    ('model_3', None),
    ('model_4', None),
    ('model_5', None),
    ('model_6', None),
    ('model_7', None),
    ('model_8', None),
    ('model_9', None),
    ('model_10', None),             # 146 epochs
    ('model_11', '#74FC74'),        # aka map estimate
    ('model_12', '#FCF567'),
    ('model_13', '#FCF567'),
    ('model_14', '#FCF567'),
    ('model_15', '#FCF567'),
    ('model_16', None),
    ('model_17', None),
    ('model_18', None),
    ('model_19', None),
    ('model_20', '#FCAC67'),
    ('model_21', '#FCAC67'),
    ('model_22', '#FCAC67'),
    ('model_23', '#FCAC67'),
    ('model_24', None),
    ('model_25', '#FCAC67'),
    ('model_26', '#FCAC67'),
    ('ml_est', '#FC6C67'),          # ml estimation
    ('ml_ensemble', '#FC6C67'),     # ensemble of 5 ml estimators
    ('model_30', '#FCAC67'),
    ('model_31', '#FCAC67'),
    ('model_28', '#FCAC67'),
    ('model_27', '#FCAC67'),
    ('model_29', '#FCAC67'),
    ('model_33', '#FCAC67'),
    ('model_34', '#FCAC67')
]

column_names = [
    'experiment_name',
    'dataset',
    'batch_size',
    'net_arc',
    'use_var_prior',
    'alpha',
    'n_particles',
    'use_latent',
    'n_hidden_dims',
    'n_epochs',
    'h_type',
    'kernel_type',
    'p',
    'move_theta_0',
    'init_theta_0',
    'Loss (Train)',
    'Loss (Test)',
    'Loss (Test (Mean (All)))',
    'Loss (Test (Mean (n_prev)))',
    'Accuracy (Train)',
    'Accuracy (Test)',
    'Accuracy (Test (Mean (All)))',
    'Accuracy (Test (Mean (n_prev)))',
    'Best Loss (Train)',
    'Best Loss (Test)',
    'Best Loss (Test (Mean (All)))',
    'Best Loss (Test (Mean (n_prev)))',
    'Best Accuracy (Train)',
    'Best Accuracy (Test)',
    'Best Accuracy (Test (Mean (All)))',
    'Best Accuracy (Test (Mean (n_prev)))',
    'Comment'
]

squeezed_column_names = [
    'experiment_name',
    'n_particles',
    'use_latent',
    'n_hidden_dims',
    'move_theta_0',
    'init_theta_0',
    'Accuracy (Test)',
    'Best Accuracy (Test)',
    'Accuracy (Test (Mean (All)))',
    'Best Accuracy (Test (Mean (All)))',
    'Accuracy (Test (Mean (n_prev)))',
    'Best Accuracy (Test (Mean (n_prev)))',
    'Comment'
]


def to_field(name):
    return name.replace(' ', '_').replace('(', '').replace(')', '')


field_names = [to_field(name) for name in column_names]

Result = namedtuple('Result', field_names, verbose=False)
data = []

# Get data from config and log files
for exp_name, _ in experiments:
    log_file_name = './Logs/' + exp_name + '.txt'
    averaged = False
    if exp_name.find('model') >= 0:
        averaged = True
    config = importlib.import_module('Configs.config_' + exp_name)
    epochs = config.n_epochs

    l_tr, l_t, l_tm, l_tmn, bl_tr, bl_t, bl_tm, bl_tmn = 8 * [1e10]
    a_tr, a_t, a_tm, a_tmn, ba_tr, ba_t, ba_tm, ba_tmn = 8 * [0.]
    with open(log_file_name) as fp:
        # skip \r and capture
        _, _ = fp.readline(),  fp.readline()
        for cnt in range(epochs):
            line_1 = fp.readline()
            line_2 = fp.readline()
            line_3 = fp.readline()
            if line_1 == '' or line_2 == '' or line_3 == '':
                break
            if averaged:
                l_tr, l_t, l_tm, l_tmn = map(float, line_2[line_2.find(':') + 2:].split('/'))
                a_tr, a_t, a_tm, a_tmn = map(float, line_3[line_3.find(':') + 2:].split('/'))
                l_tm = 1e10 if l_tm < 1e-5 else l_tm
                l_tmn = 1e10 if l_tmn < 1e-5 else l_tmn
                bl_tr, bl_t, bl_tm, bl_tmn = min(bl_tr, l_tr), min(bl_t, l_t), min(bl_tm, l_tm), min(bl_tmn, l_tmn)
                ba_tr, ba_t, ba_tm, ba_tmn = max(ba_tr, a_tr), max(ba_t, a_t), max(ba_tm, a_tm), max(ba_tmn, a_tmn)
            else:
                l_tr, l_t, l_tm, l_tmn = *map(float, line_2[line_2.find(':') + 2:].split('/')), None, None
                a_tr, a_t, a_tm, a_tmn = *map(float, line_3[line_3.find(':') + 2:].split('/')), None, None
                bl_tr, bl_t, bl_tm, bl_tmn = min(bl_tr, l_tr), min(bl_t, l_t), None, None
                ba_tr, ba_t, ba_tm, ba_tmn = max(ba_tr, a_tr), max(ba_t, a_t), None, None
        data.append(Result(config.experiment_name,
                           config.dataset,
                           config.batch_size,
                           config.net_arc,
                           config.use_var_prior,
                           config.alpha,
                           config.n_particles,
                           config.use_latent,
                           config.n_hidden_dims,
                           config.n_epochs,
                           config.h_type,
                           config.kernel_type,
                           config.p,
                           config.move_theta_0,
                           config.init_theta_0,
                           l_tr, l_t, l_tm, l_tmn,
                           a_tr, a_t, a_tm, a_tmn,
                           bl_tr, bl_t, bl_tm, bl_tmn,
                           ba_tr, ba_t, ba_tm, ba_tmn,
                           config.comment
                           )
                    )

# Iterate over the data and write it out row by row.
for idx, result in enumerate(data):
    if experiments[idx][1] is not None:
        colorize = workbook.add_format({'bg_color': experiments[idx][1]})
    else:
        colorize = workbook.add_format({})

    if experiments[idx][1] is not None:
        right_align = workbook.add_format({'align': 'right', 'bg_color': experiments[idx][1]})
    else:
        right_align = workbook.add_format({'align': 'right'})

    for jdx, name in enumerate(column_names):
        if type(getattr(result, to_field(name))) == bool:
            bool_str = str(getattr(result, to_field(name))).upper()
            worksheets[0].write(idx + 1, jdx, bool_str, right_align)
        else:
            worksheets[0].write(idx + 1, jdx, getattr(result, to_field(name)), colorize)
    for jdx, name in enumerate(squeezed_column_names):
        if type(getattr(result, to_field(name))) == bool:
            bool_str = str(getattr(result, to_field(name))).upper()
            worksheets[1].write(idx + 1, jdx, bool_str, right_align)
        else:
            worksheets[1].write(idx + 1, jdx, getattr(result, to_field(name)), colorize)

# Write head of the table
for idx, name in enumerate(column_names):
    worksheets[0].set_column(idx, idx, len(name) + 1)
    worksheets[0].write(0, idx, name)
for idx, name in enumerate(squeezed_column_names):
    worksheets[1].set_column(idx, idx, len(name) + 1)
    worksheets[1].write(0, idx, name)

workbook.close()
