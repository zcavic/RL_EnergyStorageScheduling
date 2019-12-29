import pandapower as pp

def create_network():
    network = pp.create_empty_network()

    sn_vn_transformer_data = {"sn_mva": 1, "vn_hv_kv": 20, "vn_lv_kv": 0.4, "vk_percent": 5, "vkr_percent": 1.1,
                                "pfe_kw": 1.95, "i0_percent": 0.27, "shift_degree": 0}
    pp.create_std_type(network, sn_vn_transformer_data, "SN/NN 1MVA", element='trafo')

    slack_bus = pp.create_bus(network, vn_kv=110, name="Slack Bus")

    busNodes = []
    lowVoltageBusNodes = []

    pp.create_ext_grid(network, bus=slack_bus, vm_pu=1.01, name="Grid Connection")
    mediumVoltageBusNode = pp.create_bus(network, vn_kv=20, name="MV slack side")
    pp.create_transformer(network, hv_bus=slack_bus, lv_bus=mediumVoltageBusNode, std_type="40 MVA 110/20 kV", name="VN/SN Transformer")

    for i in range(0, 100):
        busNodes.append(pp.create_bus(network, vn_kv=20, name="Bus_" + str(i+1)))
        lowVoltageBusNodes.append(pp.create_bus(network, vn_kv=0.4, name="LowVoltageBus_" + str(i+1)))
        pp.create_load(network, bus=lowVoltageBusNodes[i], p_mw=0.14, q_mvar=0.05, name="Load_" + str(i+1))
        pp.create_transformer(network, hv_bus=busNodes[i], lv_bus=lowVoltageBusNodes[i], std_type="SN/NN 1MVA", name="Transformer_" + str(i+1))

    pp.create_line(network, from_bus=mediumVoltageBusNode, to_bus=busNodes[0], length_km=0.2, name="Line_0", std_type="NA2XS2Y 1x150 RM/25 12/20 kV")

    for i in range(0, 99):
        pp.create_line(network, from_bus=busNodes[i], to_bus=busNodes[i+1], length_km=0.6, name="Line_" + str(i+1), std_type="NA2XS2Y 1x150 RM/25 12/20 kV")

    # Add capacitors with regulating switches
    pp.create_bus(network, vn_kv=20, name="Bus_Cap1")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_13'), pp.get_element_index(network, "bus", 'Bus_Cap1'), et="b", closed=False, type="LBS", name="CapSwitch1")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap1'), 0.125, 0, name="Cap1")

    pp.create_bus(network, vn_kv=20, name="Bus_Cap2")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_39'), pp.get_element_index(network, "bus", 'Bus_Cap2'), et="b", closed=False, type="LBS", name="CapSwitch2")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap2'), 0.8, 0, name="Cap2")

    pp.create_bus(network, vn_kv=20, name="Bus_Cap3")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_85'), pp.get_element_index(network, "bus", 'Bus_Cap3'), et="b", closed=False, type="LBS", name="CapSwitch3")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap3'), 0.125, 0, name="Cap3")

    pp.create_bus(network, vn_kv=20, name="Bus_Cap4")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_28'), pp.get_element_index(network, "bus", 'Bus_Cap4'), et="b", closed=False, type="LBS", name="CapSwitch4")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap4'), 3, 0, name="Cap4")

    pp.create_bus(network, vn_kv=20, name="Bus_Cap5")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_59'), pp.get_element_index(network, "bus", 'Bus_Cap5'), et="b", closed=False, type="LBS", name="CapSwitch5")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap5'), 0.25, 0, name="Cap5")

    pp.create_bus(network, vn_kv=20, name="Bus_Cap6")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_70'), pp.get_element_index(network, "bus", 'Bus_Cap6'), et="b", closed=False, type="LBS", name="CapSwitch6")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap6'), 0.8, 0, name="Cap6")

    pp.create_bus(network, vn_kv=20, name="Bus_Cap7")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_48'), pp.get_element_index(network, "bus", 'Bus_Cap7'), et="b", closed=False, type="LBS", name="CapSwitch7")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap7'), 0.25, 0, name="Cap7")

    pp.create_bus(network, vn_kv=20, name="Bus_Cap8")
    pp.create_switch(network, pp.get_element_index(network, "bus", 'Bus_95'), pp.get_element_index(network, "bus", 'Bus_Cap8'), et="b", closed=False, type="LBS", name="CapSwitch8")
    pp.create_shunt_as_capacitor(network, pp.get_element_index(network, "bus", 'Bus_Cap8'), 0.25, 0, name="Cap8")

    return network