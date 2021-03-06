
from configparser import ConfigParser
import modules.common_params as g
import requests
import progressbar as pb
import os
import cv2
import re
import ast
import traceback
import pyzm.helpers.utils as pyzmutils

g.config = {}

def str2tuple(str):
    m =  [tuple(map(int, x.strip().split(','))) for x in str.split(' ')]
    if len(m) < 3:
        raise ValueError ('{} formed an invalid polygon. Needs to have at least 3 points'.format(m))
    else:
        return m

# credit: https://stackoverflow.com/a/5320179
def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def check_and_import_zones(api):
    for mid in g.monitor_config:
        if g.monitor_config[mid].get('import_zm_zones') == 'no':
            g.logger.Debug(4,'Not importing zones for monitor:{} as the monitor specific section says no'.format(mid))
            continue
        elif g.config['import_zm_zones'] == 'no':
            g.logger.Debug(4,'Not importing zones for monitor:{} as the global setting says no and there is no local override'.format(mid))
            continue
        url = '{}/api/zones/forMonitor/{}.json'.format(g.config.get('portal'),mid)        
        g.logger.Debug(2,'Importing Zones for Monitor:{}'.format(mid))
        j = api._make_request(url=url, type='get')
        for item in j.get('zones'):
        #print ('********* ITEM TYPE {}'.format(item['Zone']['Type']))
            if item['Zone']['Type'] == 'Inactive':
                g.logger.Debug(2, 'Skipping {} as it is inactive'.format(item['Zone']['Name']))
                continue
         
            item['Zone']['Name'] = item['Zone']['Name'].replace(' ','_').lower()
            g.logger.Debug(2,'importing zoneminder polygon: {} [{}]'.format(item['Zone']['Name'], item['Zone']['Coords']))
            g.monitor_polygons[mid].append({
                'name': item['Zone']['Name'],
                'value': str2tuple(item['Zone']['Coords']),
                'pattern': None

            })
        # Now copy over pending zone patterns from process_config
        for poly in g.monitor_polygons[mid]:
            for zone_name in g.monitor_zone_patterns[mid]:
                if poly['name'] == zone_name:
                    poly['pattern'] = g.monitor_zone_patterns[mid][zone_name]
                    g.logger.Debug(2, 'replacing match pattern for polygon:{} with: {}'.format( poly['name'],poly['pattern'] ))

def convert_config_to_ml_sequence():
    ml_options={}

    for ds in g.config['detection_sequence']:
        if ds == 'object':
        
            ml_options['object'] = {
                'general':{
                    'pattern': g.config['object_detection_pattern'],
                    'disable_locks': g.config['disable_locks'],
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
        elif t == 'eval'  or t == 'dict':
            return ast.literal_eval(val) if val else None
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
        config_file = ConfigParser(interpolation=None, inline_comment_prefixes='#')
        config_file.read(args['config'])
        
        g.config['pyzm_overrides'] = {}
        if config_file.has_option('general', 'pyzm_overrides'):
            pyzm_overrides = config_file.get('general', 'pyzm_overrides')
            g.config['pyzm_overrides'] =  ast.literal_eval(pyzm_overrides) if pyzm_overrides else {}
            if args.get('debug'):
                g.config['pyzm_overrides']['dump_console'] = True
                g.config['pyzm_overrides']['log_debug'] = True
                g.config['pyzm_overrides']['log_level_debug'] = 5
                g.config['pyzm_overrides']['log_debug_target'] = None

        if config_file.has_option('general', 'use_zm_logs'):
            use_zm_logs = config_file.get('general', 'use_zm_logs')
            if use_zm_logs == 'yes':
                try:
                    import pyzm.ZMLog as zmlog
                    zmlog.init(name='zm_mlapi',override=g.config['pyzm_overrides'])
                except Exception as e:
                    g.logger.Error ('Not able to switch to ZM logs: {}'.format(e))
                else:
                    g.log = zmlog
                    g.logger=g.log
                    g.logger.Info('Switched to ZM logs')
        

        if config_file.has_option('general','secrets'):
            secrets_filename = config_file.get('general', 'secrets')
            g.config['secrets'] = secrets_filename
            g.logger.Debug (1,'secret filename: {}'.format(secrets_filename))
            has_secrets = True
            secrets_file = ConfigParser(interpolation = None, inline_comment_prefixes='#')
            try:
                with open(secrets_filename) as f:
                    secrets_file.read_file(f)
            except:
                raise            
        else:
            g.logger.Debug (1,'No secrets file configured')
        # now read config values
    
        g.polygons = []
        # first, fill in config with default values
        for k,v in g.config_vals.items():
            val = v.get('default', None)
            g.config[k] = _correct_type(val, v['type'])
            #print ('{}={}'.format(k,g.config[k]))
        
       
        # now iterate the file
        for sec in config_file.sections():
            if sec == 'secrets':
                continue
            
            # Move monitor specific stuff to a different structure
            if sec.lower().startswith('monitor-'):
                ts = sec.split('-')
                if len(ts) != 2:
                    g.logger.Error('Skipping section:{} - could not derive monitor name. Expecting monitor-NUM format')
                    continue 

                mid = int(ts[1])
                g.logger.Debug (2,'Found monitor specific section for monitor: {}'.format(mid))

                g.monitor_polygons[mid] = []
                g.monitor_config[mid] = {}
                g.monitor_zone_patterns[mid] = {}
                # Copy the sequence into each monitor because when we do variable subs
                # later, we will use this for monitor specific work
                try:
                    ml = config_file.get('ml', 'ml_sequence')
                    g.monitor_config[mid]['ml_sequence']=ml
                except:
                    g.logger.Debug (4, 'ml sequence not found in globals')
                     
                try:
                    ss = config_file.get('ml', 'stream_sequence')
                    g.monitor_config[mid]['stream_sequence']=ss
                except:
                    g.logger.Debug (4, 'stream sequence not found in globals')

                for item in config_file[sec].items():
                    k = item[0]
                    v = item[1]
                    if k.endswith('_zone_detection_pattern'):
                        zone_name = k.split('_zone_detection_pattern')[0]
                        g.logger.Debug(2, 'found zone specific pattern:{} storing'.format(zone_name))
                        g.monitor_zone_patterns[mid][zone_name] = v
                        continue
                    else:
                        if k in g.config_vals:
                        # This means its a legit config key that needs to be overriden
                            g.logger.Debug(4,'[{}] overrides key:{} with value:{}'.format(sec, k, v))
                            g.monitor_config[mid][k]=_correct_type(v,g.config_vals[k]['type'])
                           # g.monitor_config[mid].append({ 'key':k, 'value':_correct_type(v,g.config_vals[k]['type'])})
                        else:
                            if k.startswith(('object_','face_', 'alpr_')):
                                g.logger.Debug(2,'assuming {} is an ML sequence'.format(k))
                                g.monitor_config[mid][k] = v
                            else:
                                try:
                                    g.monitor_polygons[mid].append({'name': k, 'value': str2tuple(v),'pattern': None})
                                    g.logger.Debug(2,'adding polygon: {} [{}]'.format(k, v ))
                                except Exception as e:
                                    g.logger.Debug(2,'{} is not a polygon, adding it as unknown string key'.format(k))
                                    g.monitor_config[mid][k]=v

            
                            # TBD only_triggered_zones

            # Not monitor specific stuff
            else: 
                for (k, v) in config_file.items(sec):
                    if k in g.config_vals:
                        _set_config_val(k,g.config_vals[k] )
                    else:
                        #g.logger.Debug(4, 'storing unknown attribute {}={}'.format(k,v))
                        g.config[k] = v 
                        #_set_config_val(k,{'section': sec, 'default': None, 'type': 'string'} )

        


        # Parameter substitution

        g.logger.Debug (4,'Doing parameter substitution for globals')
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

        g.logger.Debug (4,'Doing parameter substitution for monitor specific entities')
        p = r'{{(\w+?)}}'
        for mid in g.monitor_config:
            for key in g.monitor_config[mid]:
                #input ('Continue')
                #print(f"PROCESSING {gk} {gv}")
                gk = key
                gv = g.monitor_config[mid][key]
                gv = '{}'.format(gv)
                #if not isinstance(gv, str):
                #    continue
                while True:
                    matches = re.findall(p,gv)
                    replaced = False
                    for match_key in matches:
                        if match_key in g.monitor_config[mid]:
                            replaced = True
                            new_val =gv.replace('{{' + match_key + '}}',str(g.monitor_config[mid][match_key]))
                            gv = new_val
                            g.monitor_config[mid][key] = gv 
                        elif match_key in g.config:
                            replaced = True
                            new_val =gv.replace('{{' + match_key + '}}',str(g.config[match_key]))
                            gv = new_val
                            g.monitor_config[mid][key] = gv
                        else:
                            g.logger.Debug(4, 'substitution key: {} not found'.format(match_key))
                    if not replaced:
                        break
            
            secrets = pyzmutils.read_config(g.config['secrets'])
            #g.monitor_config[mid]['ml_sequence'] = pyzmutils.template_fill(input_str=g.monitor_config[mid]['ml_sequence'], config=None, secrets=secrets._sections.get('secrets'))
            #g.monitor_config[mid]['ml_sequence'] = ast.literal_eval(g.monitor_config[mid]['ml_sequence'])

            #g.monitor_config[mid]['stream_sequence'] = pyzmutils.template_fill(input_str=g.monitor_config[mid]['stream_sequence'], config=None, secrets=secrets._sections.get('secrets'))
            #g.monitor_config[mid]['stream_sequence'] = ast.literal_eval(g.monitor_config[mid]['stream_sequence'])


        #print ("GLOBALS={}".format(g.config))
        #print ("\n\nMID_SPECIFIC={}".format(g.monitor_config))
        #print ("\n\nMID POLYPATTERNS={}".format(g.monitor_polypatterns))
        #print ('FINAL POLYS={}'.format(g.monitor_polygons))         
        #exit(0) 
    except Exception as e:
        g.logger.Error('Error parsing config:{}'.format(args['config']))
        g.logger.Error('Error was:{}'.format(e))
        g.logger.Fatal('error: Traceback:{}'.format(traceback.format_exc()))
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
