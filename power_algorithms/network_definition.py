import pandapower as pp
import pandapower.networks as pn

def create_network():
    network = pp.create_empty_network()

    sn_vn_transformer_data = {"sn_mva": 1, "vn_hv_kv": 20, "vn_lv_kv": 0.4, "vk_percent": 5, "vkr_percent": 1.1,
                                "pfe_kw": 1.95, "i0_percent": 0.27, "shift_degree": 0}
    pp.create_std_type(network, sn_vn_transformer_data, "SN/NN 1MVA", element='trafo')

    slack_bus = pp.create_bus(network, vn_kv=110, name="Slack Bus")

    busNodes = []
    lowVoltageBusNodes = []

    # Source
    pp.create_ext_grid(network, bus=slack_bus, vm_pu=1.00, name="Grid Connection")
    mediumVoltageBusNode = pp.create_bus(network, vn_kv=20, name="MV slack side")
    pp.create_transformer(network, hv_bus=slack_bus, lv_bus=mediumVoltageBusNode, std_type="40 MVA 110/20 kV", name="VN/SN Transformer")

    # Buses
    for i in range(0, 3):
        busNodes.append(pp.create_bus(network, vn_kv=20, name="Bus_" + str(i+1)))
        #todo od svih nam je trenutno potreban samo lowVoltageBusNodes[2], ali neka ga za sada zbog lakseg indeksiranja
        lowVoltageBusNodes.append(pp.create_bus(network, vn_kv=0.4, name="LowVoltageBus_" + str(i+1)))

    # Load in node 3
    pp.create_load(network, bus=lowVoltageBusNodes[2], p_mw=6, q_mvar=1, name="Load_" + str(2+1))
    pp.create_transformer(network, hv_bus=busNodes[2], lv_bus=lowVoltageBusNodes[2], std_type="SN/NN 1MVA", name="Transformer_" + str(2+1))

    # Solar generator in node 2
    # sgen -> static generator (PQ, not PV)
    pp.create_sgen(network, bus=busNodes[1], p_mw=6, q_mvar=1, name="Gen_" + str(2+1))

    # Storage in node 3
    # p > 0 - charging
    # p < 0 - discharging
    pp.create_storage(network, bus=busNodes[2], p_mw = 6, max_e_mwh = 1000000, soc_percent = 0.5, min_e_mwh = 0)

    # Lines
    pp.create_line(network, from_bus=mediumVoltageBusNode, to_bus=busNodes[0], length_km=0.5, name="Line_0", std_type="NA2XS2Y 1x150 RM/25 12/20 kV")

    for i in range(0, 2):
        pp.create_line(network, from_bus=busNodes[i], to_bus=busNodes[i+1], length_km=1, name="Line_" + str(i+1), std_type="NA2XS2Y 1x150 RM/25 12/20 kV")
	
    return network


def create_cigre_network_mv():
    power_grid = pn.create_cigre_network_mv(with_der="all")

    for index in power_grid.storage.index:
        power_grid.storage.p_mw.loc[index] = 5

    return power_grid
