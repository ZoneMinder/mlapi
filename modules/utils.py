
from configparser import ConfigParser
import modules.common_params as g
import requests
import progressbar as pb
import os
import cv2
import re

g.config = {}


def convert_config_to_ml_sequence():
    ml_options={}
    for ds in g.config['detection_sequence']:
        if ds == 'object':
        
            ml_options['object'] = {
                'general':{
                    'pattern': g.config['object_detection_pattern'],
                    'same_model_sequence_strategy': 'first' # 'first' 'most', 'most_unique'

                },
                'sequence': [{
                    'tpu_max_processes': g.config['tpu_max_processes'],
                    'tpu_max_lock_wait': g.config['tpu_max_lock_wait'],
                    'gpu_max_processes': g.config['gpu_max_processes'],
                    'gpu_max_lock_wait': g.config['gpu_max_lock_wait'],
                    'cpu_max_processes': g.config['cpu_max_processes'],
                    'cpu_max_lock_wait': g.config['cpu_max_lock_wait'],
                    'max_detection_size': g.config['max_detection_size'],
                    'object_config':g.config['object_config'],
                    'object_weights':g.config['object_weights'],
                    'object_labels': g.config['object_labels'],
                    'object_min_confidence': g.config['object_min_confidence'],
                    'object_framework':g.config['object_framework'],
                    'object_processor': g.config['object_processor'],
                }]
            }
        elif ds == 'face':
            ml_options['face'] = {
                'general':{
                    'pattern': g.config['face_detection_pattern'],
                    'same_model_sequence_strategy': 'first',
                #    'pre_existing_labels':['person'],
                },
                'sequence': [{
                    'tpu_max_processes': g.config['tpu_max_processes'],
                    'tpu_max_lock_wait': g.config['tpu_max_lock_wait'],
                    'gpu_max_processes': g.config['gpu_max_processes'],
                    'gpu_max_lock_wait': g.config['gpu_max_lock_wait'],
                    'cpu_max_processes': g.config['cpu_max_processes'],
                    'cpu_max_lock_wait': g.config['cpu_max_lock_wait'],
                    'face_detection_framework': g.config['face_detection_framework'],
                    'face_recognition_framework': g.config['face_recognition_framework'],
                    'face_processor': g.config['face_processor'],
                    'known_images_path': g.config['known_images_path'],
                    'face_model': g.config['face_model'],
                    'face_train_model':g.config['face_train_model'],
                    'unknown_images_path': g.config['unknown_images_path'],
                    'unknown_face_name': g.config['unknown_face_name'],
                    'save_unknown_faces': g.config['save_unknown_faces'],
                    'save_unknown_faces_leeway_pixels': g.config['save_unknown_faces_leeway_pixels'],
                    'face_recog_dist_threshold': g.config['face_recog_dist_threshold'],
                    'face_num_jitters': g.config['face_num_jitters'],
                    'face_upsample_times':g.config['face_upsample_times']
                }]

            }
        elif ds == 'alpr':
            ml_options['alpr'] = {
                'general':{
                    'pattern': g.config['alpr_detection_pattern'],
                    'same_model_sequence_strategy': 'first',
                #    'pre_existing_labels':['person'],
                },
                'sequence': [{
                    'tpu_max_processes': g.config['tpu_max_processes'],
                    'tpu_max_lock_wait': g.config['tpu_max_lock_wait'],
                    'gpu_max_processes': g.config['gpu_max_processes'],
                    'gpu_max_lock_wait': g.config['gpu_max_lock_wait'],
                    'cpu_max_processes': g.config['cpu_max_processes'],
                    'cpu_max_lock_wait': g.config['cpu_max_lock_wait'],
                    'alpr_service': g.config['alpr_service'],
                    'alpr_url': g.config['alpr_url'],
                    'alpr_key': g.config['alpr_key'],
                    'alpr_api_type': g.config['alpr_api_type'],
                    'platerec_stats': g.config['platerec_stats'],
                    'platerec_regions': g.config['platerec_regions'],
                    'platerec_min_dscore': g.config['platerec_min_dscore'],
                    'platerec_min_score': g.config['platerec_min_score'],
                    'openalpr_recognize_vehicle': g.config['openalpr_recognize_vehicle'],
                    'openalpr_country': g.config['openalpr_country'],
                    'openalpr_state': g.config['openalpr_state'],
                    'openalpr_min_confidence': g.config['openalpr_min_confidence'],
                    'openalpr_cmdline_binary': g.config['openalpr_cmdline_binary'],
                    'openalpr_cmdline_params': g.config['openalpr_cmdline_params'],
                    'openalpr_cmdline_min_confidence': g.config['openalpr_cmdline_min_confidence'],
                }]

            }
    ml_options['general'] =   {
            'model_sequence': ','.join(str(e) for e in g.config['detection_sequence'])
            #'model_sequence': 'object,face',        
    }
    if g.config['detection_mode'] == 'all':
        g.logger.Debug(3, 'Changing detection_mode from all to most_models to adapt to new features')
        g.config['detection_mode'] = 'most_models'
    return ml_options


def str_split(my_str):
    return [x.strip() for x in my_str.split(',')]

def process_config(args):
# parse config file into a dictionary with defaults

    g.config = {}
    has_secrets = False
    secrets_file = None

    def _correct_type(val,t):
        if t == 'int':
             return int(val)
        elif t == 'eval':
            return eval(val) if val else None
        elif t == 'str_split':
            return str_split(val) if val else None
        elif t  == 'string':
            return val
        elif t == 'float':
            return float(val)
        else:
            g.logger.Error ('Unknown conversion type {} for config key:{}'.format(e['type'], e['key']))
            return val

    def _set_config_val(k,v):
    # internal function to parse all keys
        val = config_file[v['section']].get(k,v['default'])

        if val and val[0] == '!': # its a secret token, so replace
            g.logger.Debug (1,'Secret token found in config: {}'.format(val));
            if not has_secrets:
                raise ValueError('Secret token found, but no secret file specified')
            if secrets_file.has_option('secrets', val[1:]):
                vn = secrets_file.get('secrets', val[1:])
                #g.logger.Debug (1,'Replacing {} with {}'.format(val,vn))
                val = vn
            else:
                raise ValueError ('secret token {} not found in secrets file {}'.format(val,secrets_filename))


        g.config[k] = _correct_type(val, v['type'])
        if k.find('password') == -1:
            dval = g.config[k]
        else:
            dval = '***********'
        #g.logger.Debug (1,'Config: setting {} to {}'.format(k,dval))

    # main        
    try:
        config_file = ConfigParser(interpolation=None)
        config_file.read(args['config'])
        

        if config_file.has_option('general','secrets'):
            secrets_filename = config_file.get('general', 'secrets')
            g.config['secrets'] = secrets_filename
            g.logger.Debug (1,'secret filename: {}'.format(secrets_filename))
            has_secrets = True
            secrets_file = ConfigParser(interpolation = None)
            try:
                with open(secrets_filename) as f:
                    secrets_file.read_file(f)
            except:
                raise            
        else:
            g.logger.Debug (1,'No secrets file configured')
        # now read config values
    
        # first, fill in config with default values
        for k,v in g.config_vals.items():
           
            g.config[k] = v.get('default', None)
            #print ('{}={}'.format(k,g.config[k]))
            
        # now iterate the file
        for sec in config_file.sections():
            if sec == 'secrets':
                continue
            for (k, v) in config_file.items(sec):
                if g.config_vals.get(k):
                    _set_config_val(k,g.config_vals[k] )
                else:
                    #g.logger.Debug(4, 'storing unknown attribute {}={}'.format(k,v))
                    g.config[k] = v 
                    #_set_config_val(k,{'section': sec, 'default': None, 'type': 'string'} )
        
         # Now lets make sure we take care of parameter substitutions {{}}
        g.logger.Debug (4,'Finally, doing parameter substitution')


        p = r'{{(\w+?)}}'
        for gk, gv in g.config.items():
            #input ('Continue')
            #print(f"PROCESSING {gk} {gv}")
            gv = '{}'.format(gv)
            #if not isinstance(gv, str):
            #    continue
            while True:
                matches = re.findall(p,gv)
                replaced = False
                for match_key in matches:
                    if match_key in g.config:
                        replaced = True
                        new_val = g.config[gk].replace('{{' + match_key + '}}',str(g.config[match_key]))
                        g.config[gk] = new_val
                        gv = new_val
                    else:
                        g.logger.Debug(4, 'substitution key: {} not found'.format(match_key))
                if not replaced:
                    break
                    
    except Exception as e:
        g.logger.Error('Error parsing config:{}'.format(args['config']))
        g.logger.Error('Error was:{}'.format(e))
        exit(0)





def draw_bbox(img, bbox, labels, classes, confidence, color=None, write_conf=True):

   # g.logger.Debug (1,"DRAW BBOX={} LAB={}".format(bbox,labels))
    slate_colors = [ 
            (39, 174, 96),
            (142, 68, 173),
            (0,129,254),
            (254,60,113),
            (243,134,48),
            (91,177,47)
        ]
   
    arr_len = len(bgr_slate_colors)
    for i, label in enumerate(labels):
        #=g.logger.Debug (1,'drawing box for: {}'.format(label))
        color = bgr_slate_colors[i % arr_len]
        if write_conf and confidence:
            label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
       
        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)

        # write text 
        font_scale = 0.8
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        #cv2.getTextSize(text, font, font_scale, thickness)
        text_size = cv2.getTextSize(label, font_type, font_scale , font_thickness)[0]
        text_width_padded = text_size[0] + 4
        text_height_padded = text_size[1] + 4

        r_top_left = (bbox[i][0], bbox[i][1] - text_height_padded)
        r_bottom_right = (bbox[i][0] + text_width_padded, bbox[i][1])
  
        cv2.putText(img, label, (bbox[i][0] + 2, bbox[i][1] - 2), font_type, font_scale, [255, 255, 255], font_thickness)

    return img
