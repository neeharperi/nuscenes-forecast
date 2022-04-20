# nuScenes dev-kit.
# Code written by Oscar Beijbom and Varun Bankiti, 2019.

def getDetectionNames(cohort_analysis=False):
    if not cohort_analysis:
        #DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
        #                'traffic_cone', 'barrier']
        DETECTION_NAMES = ['car', 'pedestrian']
        #DETECTION_NAMES = ['car']


    else:
        #DETECTION_NAMES = ['static_car', 'static_truck', 'static_bus', 'static_trailer', 'static_construction_vehicle', 'static_pedestrian', 'static_motorcycle', 'static_bicycle', 'static_traffic_cone', 'static_barrier',
        #'linear_car', 'linear_truck', 'linear_bus', 'linear_trailer', 'linear_construction_vehicle', 'linear_pedestrian', 'linear_motorcycle', 'linear_bicycle', 'linear_traffic_cone', 'linear_barrier',
        #'nonlinear_car', 'nonlinear_truck', 'nonlinear_bus', 'nonlinear_trailer', 'nonlinear_construction_vehicle', 'nonlinear_pedestrian', 'nonlinear_motorcycle', 'nonlinear_bicycle', 'nonlinear_traffic_cone', 'nonlinear_barrier']

        #DETECTION_NAMES = ['static_car', 'linear_car', 'nonlinear_car']
        #DETECTION_NAMES = ['static_car', 'moving_car']
        DETECTION_NAMES = ['static_car',  'linear_car', 'nonlinear_car', 'static_pedestrian', 'linear_pedestrian', 'nonlinear_pedestrian']
        #DETECTION_NAMES = ['static_car', 'static_pedestrian', 'moving_car', 'moving_pedestrian']


    return DETECTION_NAMES

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'trailer': 'Trailer',
                          'construction_vehicle': 'Constr. Veh.',
                          'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle',
                          'bicycle': 'Bicycle',
                          'traffic_cone': 'Traffic Cone',
                          'barrier': 'Barrier',
                          
                          'static_car': 'Static Car',
                          'static_truck': 'Static Truck',
                          'static_bus': 'Static Bus',
                          'static_trailer': 'Static Trailer',
                          'static_construction_vehicle': 'Static Constr. Veh.',
                          'static_pedestrian': 'Static Pedestrian',
                          'static_motorcycle': 'Static Motorcycle',
                          'static_bicycle': 'Static Bicycle',
                          'static_traffic_cone': 'Static Traffic Cone',
                          'static_barrier': 'Static Barrier',

                          'moving_car': 'Moving Car',
                          'moving_truck': 'Moving Truck',
                          'moving_bus': 'Moving Bus',
                          'moving_trailer': 'Moving Trailer',
                          'moving_construction_vehicle': 'Moving Constr. Veh.',
                          'moving_pedestrian': 'Moving Pedestrian',
                          'moving_motorcycle': 'Moving Motorcycle',
                          'moving_bicycle': 'Moving Bicycle',
                          'moving_traffic_cone': 'Moving Traffic Cone',
                          'moving_barrier': 'Moving Barrier',

                          'linear_car': 'Linear Car',
                          'linear_truck': 'Linear Truck',
                          'linear_bus': 'Linear Bus',
                          'linear_trailer': 'Linear Trailer',
                          'linear_construction_vehicle': 'Linear Constr. Veh.',
                          'linear_pedestrian': 'Linear Pedestrian',
                          'linear_motorcycle': 'Linear Motorcycle',
                          'linear_bicycle': 'Linear Bicycle',
                          'linear_traffic_cone': 'Linear Traffic Cone',
                          'linear_barrier': 'Linear Barrier',

                          'nonlinear_car': 'Non-Linear Car',
                          'nonlinear_truck': 'Non-Linear Truck',
                          'nonlinear_bus': 'Non-Linear Bus',
                          'nonlinear_trailer': 'Non-Linear Trailer',
                          'nonlinear_construction_vehicle': 'Non-Linear Constr. Veh.',
                          'nonlinear_pedestrian': 'Non-Linear Pedestrian',
                          'nonlinear_motorcycle': 'Non-Linear Motorcycle',
                          'nonlinear_bicycle': 'Non-Linear Bicycle',
                          'nonlinear_traffic_cone': 'Non-Linear Traffic Cone',
                          'nonlinear_barrier': 'Non-Linear Barrier',                       
                          }

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    'traffic_cone': 'C8',
                    'barrier': 'C9',

                    'static_car': 'C0',
                    'static_truck': 'C1',
                    'static_bus': 'C2',
                    'static_trailer': 'C3',
                    'static_construction_vehicle': 'C4',
                    'static_pedestrian': 'C5',
                    'static_motorcycle': 'C6',
                    'static_bicycle': 'C7',
                    'static_traffic_cone': 'C8',
                    'static_barrier': 'C9',

                    'moving_car': 'C0',
                    'moving_truck': 'C1',
                    'moving_bus': 'C2',
                    'moving_trailer': 'C3',
                    'moving_construction_vehicle': 'C4',
                    'moving_pedestrian': 'C5',
                    'moving_motorcycle': 'C6',
                    'moving_bicycle': 'C7',
                    'moving_traffic_cone': 'C8',
                    'moving_barrier': 'C9',

                    'linear_car': 'C0',
                    'linear_truck': 'C1',
                    'linear_bus': 'C2',
                    'linear_trailer': 'C3',
                    'linear_construction_vehicle': 'C4',
                    'linear_pedestrian': 'C5',
                    'linear_motorcycle': 'C6',
                    'linear_bicycle': 'C7',
                    'linear_traffic_cone': 'C8',
                    'linear_barrier': 'C9',

                    'nonlinear_car': 'C0',
                    'nonlinear_truck': 'C1',
                    'nonlinear_bus': 'C2',
                    'nonlinear_trailer': 'C3',
                    'nonlinear_construction_vehicle': 'C4',
                    'nonlinear_pedestrian': 'C5',
                    'nonlinear_motorcycle': 'C6',
                    'nonlinear_bicycle': 'C7',
                    'nonlinear_traffic_cone': 'C8',
                    'nonlinear_barrier': 'C9',                      
                    }

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped'}

TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err'] + ['avg_disp_err', 'final_disp_err', 'miss_rate'] #+ ['reverse_avg_disp_err', 'reverse_final_disp_err', 'reverse_miss_rate']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     'attr_err': 'Attr.', 'avg_disp_err' : "ADE", 'final_disp_err' : "FDE", 'miss_rate': "MR",
                    # 'reverse_avg_disp_err' : "RADE", 'reverse_final_disp_err' : "RFDE", 'reverse_miss_rate': "RMR"
                    }

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    'attr_err': '1-acc.',
                    'avg_disp_err' : 'm',
                    'final_disp_err' : 'm',
                    'miss_rate' : '1-acc',
                    #'reverse_avg_disp_err' : 'm',
                    #'reverse_final_disp_err' : 'm',
                    #'reverse_miss_rate' : '1-acc'
                    }
