from flask import Flask, render_template, request, send_file
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

################################### Fuzzifikasi ###################################
# variabel input
var_names = [
    'algoritma_dan_pemrograman', 'struktur_data', 'pemrograman_berorientasi_objek',
    'matematika_diskrit', 'statistika_dan_probabilitas', 'data_mining',
    'komunikasi_data_dan_jaringan_komputer'
]

# semesta pembicaraan untuk variabel input
algoritma_dan_pemrograman = ctrl.Antecedent(np.arange(0, 101, 1), 'algoritma_dan_pemrograman')
struktur_data = ctrl.Antecedent(np.arange(0, 101, 1), 'struktur_data')
pemrograman_berorientasi_objek = ctrl.Antecedent(np.arange(0, 101, 1), 'pemrograman_berorientasi_objek')

matematika_diskrit = ctrl.Antecedent(np.arange(0, 101, 1), 'matematika_diskrit')
statistika_dan_probabilitas = ctrl.Antecedent(np.arange(0, 101, 1), 'statistika_dan_probabilitas')
data_mining = ctrl.Antecedent(np.arange(0, 101, 1), 'data_mining')

komunikasi_data_dan_jaringan_komputer = ctrl.Antecedent(np.arange(0, 101, 1), 'komunikasi_data_dan_jaringan_komputer')

# variabel output
software_engineering = ctrl.Consequent(np.arange(0, 101, 1), 'software_engineering')
data_science = ctrl.Consequent(np.arange(0, 101, 1), 'data_science')
cyber_security = ctrl.Consequent(np.arange(0, 101, 1), 'cyber_security')

# himpunan variabel input
for var in [algoritma_dan_pemrograman, struktur_data, pemrograman_berorientasi_objek,
            matematika_diskrit, statistika_dan_probabilitas, data_mining,
            komunikasi_data_dan_jaringan_komputer]:
    var['rendah'] = fuzz.trimf(var.universe, [0, 0, 65])
    var['sedang'] = fuzz.trimf(var.universe, [50, 65, 80])
    var['tinggi'] = fuzz.trimf(var.universe, [65, 100, 100])

# himpunan variabel output
for peminatan in [software_engineering, data_science, cyber_security]:
    peminatan['rendah'] = fuzz.trimf(peminatan.universe, [0, 0, 50])
    peminatan['tinggi'] = fuzz.trimf(peminatan.universe, [50, 100, 100])

###################################### Rules ######################################

rules = [
    # Software Engineering tinggi
    ctrl.Rule(algoritma_dan_pemrograman['tinggi'] & struktur_data['tinggi'] & pemrograman_berorientasi_objek['tinggi'],
              software_engineering['tinggi']),
    ctrl.Rule(algoritma_dan_pemrograman['tinggi'] & struktur_data['tinggi'],
              software_engineering['tinggi']),
    ctrl.Rule(pemrograman_berorientasi_objek['tinggi'] & algoritma_dan_pemrograman['tinggi'],
              software_engineering['tinggi']),
    ctrl.Rule(algoritma_dan_pemrograman['tinggi'] & pemrograman_berorientasi_objek['tinggi'],
              software_engineering['tinggi']),

    # Data Science tinggi
    ctrl.Rule(statistika_dan_probabilitas['tinggi'] & matematika_diskrit['tinggi'] & data_mining['tinggi'],
              data_science['tinggi']),
    ctrl.Rule(statistika_dan_probabilitas['tinggi'] & matematika_diskrit['tinggi'],
              data_science['tinggi']),
    ctrl.Rule(data_mining['tinggi'] & matematika_diskrit['tinggi'],
              data_science['tinggi']),
    ctrl.Rule(statistika_dan_probabilitas['tinggi'] & data_mining['tinggi'],
              data_science['tinggi']),

    # Cyber Security tinggi
    ctrl.Rule(matematika_diskrit['tinggi'] & algoritma_dan_pemrograman['tinggi'] &
              komunikasi_data_dan_jaringan_komputer['tinggi'], cyber_security['tinggi']),
    ctrl.Rule(matematika_diskrit['tinggi'] & algoritma_dan_pemrograman['tinggi'],
              cyber_security['tinggi']),
    ctrl.Rule(komunikasi_data_dan_jaringan_komputer['tinggi'] & algoritma_dan_pemrograman['tinggi'],
              cyber_security['tinggi']),
    ctrl.Rule(matematika_diskrit['tinggi'] & komunikasi_data_dan_jaringan_komputer['tinggi'],
              cyber_security['tinggi']),

    # Software Engineering rendah
    ctrl.Rule(algoritma_dan_pemrograman['rendah'] & struktur_data['rendah'] & pemrograman_berorientasi_objek['rendah'],
              software_engineering['rendah']),
    ctrl.Rule(algoritma_dan_pemrograman['rendah'] & struktur_data['rendah'],
              software_engineering['rendah']),
    ctrl.Rule(pemrograman_berorientasi_objek['rendah'] & algoritma_dan_pemrograman['rendah'],
              software_engineering['rendah']),
    ctrl.Rule(algoritma_dan_pemrograman['rendah'] & pemrograman_berorientasi_objek['rendah'],
              software_engineering['rendah']),

    # Data Science rendah
    ctrl.Rule(statistika_dan_probabilitas['rendah'] & matematika_diskrit['rendah'] & data_mining['rendah'],
              data_science['rendah']),
    ctrl.Rule(statistika_dan_probabilitas['rendah'] & matematika_diskrit['rendah'],
              data_science['rendah']),
    ctrl.Rule(data_mining['rendah'] & matematika_diskrit['rendah'],
              data_science['rendah']),
    ctrl.Rule(statistika_dan_probabilitas['rendah'] & data_mining['rendah'],
              data_science['rendah']),

    # Cyber Security rendah
    ctrl.Rule(matematika_diskrit['rendah'] & algoritma_dan_pemrograman['rendah'] &
              komunikasi_data_dan_jaringan_komputer['rendah'], cyber_security['rendah']),
    ctrl.Rule(matematika_diskrit['rendah'] & algoritma_dan_pemrograman['rendah'],
              cyber_security['rendah']),
    ctrl.Rule(komunikasi_data_dan_jaringan_komputer['rendah'] & algoritma_dan_pemrograman['rendah'],
              cyber_security['rendah']),
    ctrl.Rule(matematika_diskrit['rendah'] & komunikasi_data_dan_jaringan_komputer['rendah'],
              cyber_security['rendah']),
]



################################## Defuzzyfikasi ##################################
# control system
peminatan_ctrl = ctrl.ControlSystem(rules)
peminatan_simulasi = ctrl.ControlSystemSimulation(peminatan_ctrl)

def generate_plots():
    plot_paths = {}

    for pem in [software_engineering, data_science, cyber_security]:
        pem.view(sim=peminatan_simulasi)
        plot_path = os.path.join(app.root_path, 'static', f'{pem.label}_plot.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths[pem.label] = f'{pem.label}_plot.png'

    return plot_paths

@app.route('/', methods=['GET', 'POST'])
def index():
    global var_names

    if request.method == 'POST':
        input_values = {}
        for var_name in var_names:
            input_values[var_name] = float(request.form[var_name])

        for var_name, value in input_values.items():
            peminatan_simulasi.input[var_name] = value

        peminatan_simulasi.compute()

        recommendations = {}
        if peminatan_simulasi.output['software_engineering'] > 50:
            recommendations['software_engineering'] = 'tinggi'
        else:
            recommendations['software_engineering'] = 'rendah'

        if peminatan_simulasi.output['data_science'] > 50:
            recommendations['data_science'] = 'tinggi'
        else:
            recommendations['data_science'] = 'rendah'

        if peminatan_simulasi.output['cyber_security'] > 50:
            recommendations['cyber_security'] = 'tinggi'
        else:
            recommendations['cyber_security'] = 'rendah'

        plot_paths = generate_plots()

        return render_template('hasil.html', recommendations=recommendations, plot_paths=plot_paths)
    else:
        return render_template('input.html', var_names=var_names)

@app.route('/plot/<plot_name>')
def plot(plot_name):
    plot_path = os.path.join(app.root_path, 'static', plot_name)
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
