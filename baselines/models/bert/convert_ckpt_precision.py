import os
from tensorflow import pywrap_tensorflow
import numpy as np
import tensorflow as tf
from shutil import copyfile
import argparse
import re
from tensorflow.contrib.slim import get_variables_to_restore
import pdb
from collections import OrderedDict, defaultdict
# parser
#os.environ['TF_POPLAR_FLAGS']="--use_ipu_model"

parser = argparse.ArgumentParser(description='Convert Precision Model')
parser.add_argument('--ckpt_dir', type=str, default='',
                    help='path to convert checkpoint file directory (default:'')')
parser.add_argument('--test', action="store_true",
                    help='weather split filters to test model')
parser.add_argument('--print', action="store_true",
                    help='weather print the  model parameters name')
parser.add_argument('--convert', action="store_true",
                    help='convert action')
parser.add_argument('--remove', action="store_true",
                    help='remove action')
parser.add_argument('--rename', action="store_true",
                    help='rename action')
parser.add_argument('--googletogc', action="store_true",
                    help='rename action')
parser.add_argument('--compare', action="store_true",
                    help='rename action')
parser.add_argument('--ckpt_dir_b', type=str, default='',
                    help='path to convert checkpoint file directory (default:'')')
parser.add_argument('--gctogoogle', action="store_true",
                    help='rename action')


args = parser.parse_args()

def convert_ckpt_to_fp(checkpoint_path,data_type=np.float32):
    """Convert checkpoint to fp weights and return saver.
    Args:
        init_checkpoint: Path to checkpoint file.
        data_type: np.float16, np.float32, np.float64,

    """
    ckpt = 'ckpt'
    sync_file = []
    checkpoint_name = None
    os.chdir(checkpoint_path)
    for each_file in  os.listdir(os.curdir):
        if ckpt in each_file:
            # checkpoint_name = each_file.split(ckpt)[0]+ckpt
            checkpoint_name = each_file.split(ckpt)[0]+ckpt+each_file.split(ckpt)[1].split('.')[0]
            break
        
    if checkpoint_name is None:
        return
    
    curent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/'
    out_dir = curent_dir +  checkpoint_path + "-F32"+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 

    for each_file in  os.listdir(os.curdir):
        ext = os.path.splitext(each_file)[1]
        if ext in ['.txt','.json']:
            copyfile(curent_dir+checkpoint_path+'/'+each_file, out_dir+each_file)
        
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_name)
    var_to_map = reader.get_variable_to_shape_map()
    val_f = {}
    for key, dim in var_to_map.items():
        if key == 'global_step':
            val_f[key.strip(":0")] = tf.Variable(reader.get_tensor(key),name=key)
            continue
        if 'word_embeddings' in key:
            val_f[key.strip(":0")] = tf.Variable(reader.get_tensor(key).astype(data_type),name=key)
            continue
        val_f[key.strip(":0")] = tf.Variable(reader.get_tensor(key).astype(data_type),name=key)
        if args.test:
            if 'word_embeddings' in key:
                temp = reader.get_tensor(key)[:2896,:]
                val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))#119547
            if 'dense' in key:
                if len(dim)>1:
                    need_split_dim1 = False
                    need_split_dim2 = False
                    need_split_dim1 = True if dim[0]==3072 else False
                    need_split_dim2 = True if dim[1]==3072 else False
                    if need_split_dim1:
                        temp = reader.get_tensor(key)[:2048,:]
                        val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
                    elif need_split_dim2:
                        temp = reader.get_tensor(key)[:,:2048]
                        val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
                    elif need_split_dim1 and need_split_dim2:
                        temp = reader.get_tensor(key)[:2048,:2048]
                        val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
                else:
                    if dim[0]==3072:
                        temp = reader.get_tensor(key)[:2048]
                        val_f[key.strip(":0")] =  tf.Variable(temp.astype(data_type))
        
    #get parameters before convert
    param_log_origin=''
    for key in var_to_map:
        param_log_origin += "tensor_name: "+key+"  shape:"+str(reader.get_tensor(key).shape)+"\r\n"
        param_log_origin += str(reader.get_tensor(key))+"\r\n"  
    writer = open(out_dir+'Param-'+str(reader.get_tensor(key).dtype)+'.txt', 'w', encoding="utf-8")
    writer.write(param_log_origin)      
  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # new_saver = tf.train.import_meta_graph(curent_dir + checkpoint_path +'/'+checkpoint_name+'.meta')
        # new_saver.restore(sess,curent_dir + checkpoint_path +'/'+checkpoint_name)  
        # saver = tf.train.Saver(val_f)
        saver = tf.train.Saver()
        saver.save(sess, out_dir+checkpoint_name)  

    #save parameters after convert
    reader_convert = pywrap_tensorflow.NewCheckpointReader(out_dir+checkpoint_name)
    var_to_map_convert = reader_convert.get_variable_to_shape_map()  
    param_log_convert=''
    for item in var_to_map_convert:
        param_log_convert += "tensor_name: "+item+"  shape:"+str(reader_convert.get_tensor(item).shape)+"\r\n"
        param_log_convert += str(reader_convert.get_tensor(item))+"\r\n" 
    writer = open(out_dir+'Param-'+str(reader_convert.get_tensor(item).dtype)+'.txt', 'w', encoding="utf-8")
    writer.write(param_log_convert)      
    
    print("Convert Finish!")
    print("Save to path:"+out_dir)    


def remove_train_cache_parameters(checkpoint_path):
    ckpt = 'ckpt'
    checkpoint_name = None
    os.chdir(checkpoint_path)
    for each_file in  os.listdir(os.curdir):
        if ckpt in each_file:
            # checkpoint_name = each_file.split(ckpt)[0]+ckpt
            checkpoint_name = each_file.split(ckpt)[0]+ckpt+each_file.split(ckpt)[1].split('.')[0]
            break
    if checkpoint_name is None:
        return 
    
    curent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/'
    out_dir = curent_dir +  checkpoint_path + "_without_adam"+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    

    for each_file in  os.listdir(os.curdir):
        ext = os.path.splitext(each_file)[1]
        if ext in ['.txt','.json']:
            copyfile(curent_dir+checkpoint_path+'/'+each_file, out_dir+each_file)
        
    # graph = tf.Graph()
    # with graph.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess = tf.Session()
        #checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
        #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver = tf.train.import_meta_graph(curent_dir + checkpoint_path +'/'+checkpoint_name+'.meta')
        saver.restore(sess, curent_dir + checkpoint_path +'/'+checkpoint_name)
    
        # remove relevent of adam for small storage
        variables = get_variables_to_restore()
        # other_vars = [variable for variable in variables if not re.search("adam", variable.name) and not re.search("global_step", variable.name)]
        other_vars = [variable for variable in variables if not re.search("adam", variable.name) and not re.search("Momentum",variable.name)]
        var_saver = tf.train.Saver(other_vars)
        var_saver.save(sess, out_dir+checkpoint_name)
    
    print("Convert Finish!")
    print("Save to path:"+out_dir)     


def rename_ckpt_tensor_name(checkpoint_path):
    ckpt = 'ckpt'
    checkpoint_name = None
    os.chdir(checkpoint_path)
    for each_file in  os.listdir(os.curdir):
        if ckpt in each_file:
            checkpoint_name = each_file.split(ckpt)[0]+ckpt+each_file.split(ckpt)[1].split('.')[0]
            break
    if checkpoint_name is None:
        return 
    
    curent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/'
    out_dir = curent_dir +  checkpoint_path + "_rename"+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    

    for each_file in  os.listdir(os.curdir):
        ext = os.path.splitext(each_file)[1]
        if ext in ['.txt','.json']:
            copyfile(curent_dir+checkpoint_path+'/'+each_file, out_dir+each_file)

    checkpoint_path =curent_dir + checkpoint_path+'/'+checkpoint_name
    with tf.Session() as sess:
        # pdb.set_trace()
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_path):
            print(var_name)
            if 'word_embeddings' in var_name:
                # pdb.set_trace()
                var = tf.contrib.framework.load_variable(checkpoint_path, var_name)
                var = tf.Variable(var.T, shape=var.T.shape, name=var_name)
            else:
                var = tf.contrib.framework.load_variable(checkpoint_path, var_name)
                var = tf.Variable(var, name=var_name)
            '''
            if 'cls' not in var_name:
                if 'encoder' in var_name:
                    new_var_name = var_name[13:]
                    print(new_var_name)
                    var = tf.contrib.framework.load_variable(checkpoint_path, var_name)
                    var = tf.Variable(var, name=new_var_name)
                elif 'embedding' in var_name:
                    new_var_name = var_name[16:]
                    print(new_var_name)
                    var = tf.contrib.framework.load_variable(checkpoint_path, var_name)
                    var = tf.Variable(var, name=new_var_name)
                elif 'pooler' in var_name:
                    new_var_name = var_name[5:]
                    print(new_var_name)
                    var = tf.contrib.framework.load_variable(checkpoint_path, var_name)
                    var = tf.Variable(var, name=new_var_name)
                elif 'ouput_weights' or 'output_bias' in var_name:
                    new_var_name = var_name
                    print(new_var_name)
                    var = tf.contrib.framework.load_variable(checkpoint_path, var_name)
                    var = tf.Variable(var, name=new_var_name)
            '''

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, out_dir+checkpoint_name)
    
    print("Rename Finish!")
    print("Save to path:"+out_dir)     

def print_ckpt_tensor_name(checkpoint_path):
    import pdb
    pdb.set_trace()
    num_tensor = 0
    ckpt = 'ckpt'
    checkpoint_name = None
    os.chdir(checkpoint_path)
    for each_file in  os.listdir(os.curdir):
        if ckpt in each_file:
            checkpoint_name = each_file.split(ckpt)[0]+ckpt+each_file.split(ckpt)[1].split('.')[0]
            break
        
    if checkpoint_name is None:
        return
    
    model_reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_name)
    var_dict = model_reader.get_variable_to_shape_map()
    for key in var_dict:
        num_tensor = num_tensor + 1
        print(key+" "+str(model_reader.get_tensor(key).shape))
        if key == 'bert/encoder/layer_2/attention/self/qkv_weight'\
        or key == 'bert/embeddings/word_embeddings'\
        or key == 'bert/pooler/dense/kernel':
            print(model_reader.get_tensor(key))
    print(num_tensor)

def filter_optimizer(variables_name, optimizer_names):
    flag = False
    for name in optimizer_names:
        if name.lower() in variables_name.lower():
            flag = True
    return flag

def convert_google_ckpt_to_gc(ckpt_file,
                              output_dir=None,
                              num_embed_split=1,
                              vocab_size=52000,
                              use_attention_bias=False,
                              use_qkv_bias=False,
                              use_cls_layer=False,
                              dtype=tf.float16,
                              label_num=1):
    """ Convert Google original checkpoint to GC bert checkpoint
    there are several difference between our GC bert and origin google bert:
        1. gc_bert do not have attention_probs_dropout_prob
        2. gc_bert do not have mlm projection layer
        3. gc_bert do not have attention_projection_bias
        4. rename scope `bert/encoder/layer_x/attention/output/` to `bert/encoder/layer_x/attention/projection/`
        5. combine query, key, value layer to qkv_weight and qkv_bias layer. This changes might cause different performance on lamb optimizer,
           so the optimizer has been modified.
        6. In some cases, gc_bert supports word embedding split and rename the scope to `bert/embeddings/s{i}/word_embeddings`.
    Args:
        ckpt_file: str, Google checkpoint.
        output_dir: str, Path to save converted GC checkpoint.
        num_embed_split: int, number of word embedding need to be split. Only will be used when load origin google checkpoint
        vocab_size: int, vocabulary size. GC bert cut original 30522 to 30400 for better performance.
        use_attention_bias: bool, whether to use attention bias. Defaults to False.
        use_qkv_bias: bool, whether to use bias in qkv layers. Defaults to False.
        use_cls_layer: bool, whether to use dense layer before mlm loss. Defaults to False
        dtype: tf.float32 or tf.float16, type of tensor in output ckpt file. Only will be used when load origin google checkpoint

    Returns:
        None
    """
    graph = tf.Graph()
    dir_name, ckpt_name = os.path.split(ckpt_file)
    if not output_dir:
        output_dir = os.path.join(dir_name, "gc_ckpt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.Session()
        num_hidden_layers = 0
        optimizer_names = ["adam", "Momentum", "lamb"]  # optimizer weights
        qkv_layers = defaultdict(dict)
        saved_variables = []	
        for tensor_name in var_to_shape_map:
            # Filter the optimizer variables
            if filter_optimizer(tensor_name, optimizer_names):
                continue
            if not use_cls_layer and "transform" in tensor_name:	
                # print("Abandon dense layer before mlm loss.")
                continue
            if not use_attention_bias and "output/dense/bias" in tensor_name:	
                # print("Abandon attention bias")	
                continue
            tensor_value = tf.cast(reader.get_tensor(tensor_name), dtype=dtype)
            if "word_embeddings" in tensor_name and num_embed_split > 1:
                # split word_embeddings when num_split>1
                word_embeddings = tensor_value[:vocab_size, :]
                hidden_size = np.shape(word_embeddings)[1]
                assert vocab_size % num_embed_split == 0
                size_per_slice = int(vocab_size / num_embed_split)
                for i in range(num_embed_split):
                    start_idx = i * size_per_slice
                    end_idx = (i+1) * size_per_slice	
                    we_pieces = tf.Variable(	
                        word_embeddings[start_idx:end_idx, :],
                        shape=(size_per_slice, hidden_size),
                        name=f"bert/embeddings/s{i}/word_embeddings")	
                    saved_variables.append(we_pieces)	

            # Rename tensor
            elif "attention/output" in tensor_name:	
                new_name = tensor_name.replace("attention/output", "attention/projection")
                if "LayerNorm" in tensor_name:
                    new_name = new_name.replace("LayerNorm", "GroupNorm")
                proj = tf.Variable(tensor_value,name=new_name)
                saved_variables.append(proj)	
            elif "LayerNorm" in tensor_name:
                ln = tf.Variable(tensor_value,
                                name=tensor_name.replace("LayerNorm", "GroupNorm"))	
                saved_variables.append(ln)
            # Find query, key, value.
            elif "query" in tensor_name or \
                "key" in tensor_name or \
                    "value" in tensor_name:
                layer_idx = int(tensor_name.split(
                    "/")[2].split("_")[1])  # get the layer_{i}
                num_hidden_layers = max(layer_idx, num_hidden_layers)
                qkv_layers[layer_idx][tensor_name] = tensor_value
            else:	
                others_var = tf.Variable(tensor_value, name=tensor_name)
                saved_variables.append(others_var)	

        print("Start to combine query,key,value layers to qkv layer...")	
        for i in range(num_hidden_layers+1):
            layer_name = f"bert/encoder/layer_{i}/attention/self"
            # Combine query,key,value to qkv_weight
            layer_tensors = qkv_layers[i]
            qkv_weight = []
            qkv_bias = []
            for name in ["query", "key", "value"]:
                weight_name = layer_name + f"/{name}/kernel"
                bias_name = layer_name + f"/{name}/bias"
                qkv_weight.append(layer_tensors[weight_name])
                if use_qkv_bias:
                    qkv_bias.append(layer_tensors[bias_name])

            qkv_weight = tf.concat(qkv_weight, axis=1)
            qkv = tf.Variable(qkv_weight, shape=qkv_weight.shape,	
                            name=layer_name+"/qkv_weight")	
            saved_variables.append(qkv)

            if use_qkv_bias:
                qkv_bias = tf.concat(qkv_bias, axis=0)	
                qkv_b = tf.Variable(qkv_bias, shape=qkv_bias.shape,	
                                name=layer_name+"/qkv_bias")
                saved_variables.append(qkv_b)
            else:	
                print(f"Abandon QKV bias in layer_{i}")	
        

        #loss
        loss_weight = tf.get_variable(shape=(label_num,768),dtype=tf.float16,
                                initializer=tf.truncated_normal_initializer(stddev=0.02),	
                                name="output_weights")
        saved_variables.append(loss_weight)
        loss_bias = tf.get_variable(shape=(label_num,),dtype=tf.float16,
                                initializer=tf.zeros_initializer(),		
                                name="output_bias")
        saved_variables.append(loss_bias)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)	
        print("Save to :" + output_file)

def get_embeding(path="/home/xihuaiwen/chinese/CLUE_B/baselines/models/bert/prev_trained_model/chinese_L-12_H-768_A-12/gc_ckpt/model.ckpt-525000"):
    graph = tf.Graph()
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.Session()
        for key in var_to_shape_map:
            if "adam" not in key and "Momentum" not in key:
                if 'word_embeddings' in key:
                    val = reader.get_tensor(key)
    return val

def convert_ipu_ckpt_to_gc(ckpt_file,
                              output_dir=None,
                              num_embed_split=1,
                              vocab_size=30400,
                              use_attention_bias=False,
                              use_qkv_bias=False,
                              use_cls_layer=False,
                              dtype=tf.float16,
                              label_num=1):
    """ Convert Google original checkpoint to GC bert checkpoint
    there are several difference between our GC bert and origin google bert:
        1. gc_bert do not have attention_probs_dropout_prob
        2. gc_bert do not have mlm projection layer
        3. gc_bert do not have attention_projection_bias
        4. rename scope `bert/encoder/layer_x/attention/output/` to `bert/encoder/layer_x/attention/projection/`
        5. combine query, key, value layer to qkv_weight and qkv_bias layer. This changes might cause different performance on lamb optimizer,
           so the optimizer has been modified.
        6. In some cases, gc_bert supports word embedding split and rename the scope to `bert/embeddings/s{i}/word_embeddings`.
    Args:
        ckpt_file: str, Google checkpoint.
        output_dir: str, Path to save converted GC checkpoint.
        num_embed_split: int, number of word embedding need to be split. Only will be used when load origin google checkpoint
        vocab_size: int, vocabulary size. GC bert cut original 30522 to 30400 for better performance.
        use_attention_bias: bool, whether to use attention bias. Defaults to False.
        use_qkv_bias: bool, whether to use bias in qkv layers. Defaults to False.
        use_cls_layer: bool, whether to use dense layer before mlm loss. Defaults to False
        dtype: tf.float32 or tf.float16, type of tensor in output ckpt file. Only will be used when load origin google checkpoint

    Returns:
        None
    """
    graph = tf.Graph()
    dir_name, ckpt_name = os.path.split(ckpt_file)
    if not output_dir:
        output_dir = os.path.join(dir_name, "gc_ckpt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.Session()
        num_hidden_layers = 0
        optimizer_names = ["adam", "Momentum", "lamb"]  # optimizer weights
        qkv_layers = defaultdict(dict)
        saved_variables = []
        emb_list = []	
        for tensor_name in var_to_shape_map:
            # Filter the optimizer variables
            if filter_optimizer(tensor_name, optimizer_names):
                continue
            if not use_cls_layer and "transform" in tensor_name:	
                # print("Abandon dense layer before mlm loss.")
                continue
            if not use_attention_bias and "output/dense/bias" in tensor_name:	
                # print("Abandon attention bias")	
                continue
            tensor_value = tf.cast(reader.get_tensor(tensor_name), dtype=dtype)
            if "word_embeddings" in tensor_name:
                emb_list.append(tensor_name) 
                # split word_embeddings when num_split>1
                '''
                word_embeddings = tensor_value[:vocab_size, :]
                hidden_size = np.shape(word_embeddings)[1]
                assert vocab_size % num_embed_split == 0
                size_per_slice = int(vocab_size / num_embed_split)
                for i in range(num_embed_split):
                    start_idx = i * size_per_slice
                    end_idx = (i+1) * size_per_slice	
                    we_pieces = tf.Variable(	
                        word_embeddings[start_idx:end_idx, :],
                        shape=(size_per_slice, hidden_size),
                        name=f"bert/embeddings/s{i}/word_embeddings")	
                    saved_variables.append(we_pieces)	
                '''

            # Rename tensor
            elif "attention/output" in tensor_name:	
                new_name = tensor_name.replace("attention/output", "attention/projection")
                if "LayerNorm" in tensor_name:
                    new_name = new_name.replace("LayerNorm", "GroupNorm")
                proj = tf.Variable(tensor_value,name=new_name)
                saved_variables.append(proj)	
            elif "LayerNorm" in tensor_name:
                ln = tf.Variable(tensor_value,
                                name=tensor_name.replace("LayerNorm", "GroupNorm"))	
                saved_variables.append(ln)
            # Find query, key, value.
            elif "query" in tensor_name or \
                "key" in tensor_name or \
                    "value" in tensor_name:
                layer_idx = int(tensor_name.split(
                    "/")[2].split("_")[1])  # get the layer_{i}
                num_hidden_layers = max(layer_idx, num_hidden_layers)
                qkv_layers[layer_idx][tensor_name] = tensor_value
            else:	
                others_var = tf.Variable(tensor_value, name=tensor_name)
                saved_variables.append(others_var)	

        print("Start to combine query,key,value layers to qkv layer...")	
        print("Start to combine word embedding ...")
        word_embeddings = np.sort(emb_list)
        embeding_vals = [reader.get_tensor(key) for key in word_embeddings]
        unit_embedding = np.vstack(embeding_vals)
        

        # unit_embedding = get_embeding()
        # word_embedding = tf.concat(emb_list, axis=0)
        word = tf.Variable(unit_embedding, shape=unit_embedding.shape,	
                            name="bert/embeddings/word_embeddings",dtype=tf.float16)	
        saved_variables.append(word)
        '''
        for i in range(num_hidden_layers+1):
            layer_name = f"bert/encoder/layer_{i}/attention/self"
            # Combine query,key,value to qkv_weight
            layer_tensors = qkv_layers[i]
            qkv_weight = []
            qkv_bias = []
            for name in ["query", "key", "value"]:
                weight_name = layer_name + f"/{name}/kernel"
                bias_name = layer_name + f"/{name}/bias"
                qkv_weight.append(layer_tensors[weight_name])
                qkv_bias.append(layer_tensors[bias_name])

            qkv_weight = tf.concat(qkv_weight, axis=1)
            qkv = tf.Variable(qkv_weight, shape=qkv_weight.shape,	
                            name=layer_name+"/qkv_weight")	
            saved_variables.append(qkv)

            if use_qkv_bias:
                qkv_bias = tf.concat(qkv_bias, axis=0)	
                qkv_b = tf.Variable(qkv_bias, shape=qkv_bias.shape,	
                                name=layer_name+"/qkv_bias")
                saved_variables.append(qkv_b)
            else:	
                print(f"Abandon QKV bias in layer_{i}")	
        '''

        #loss
        loss_weight = tf.get_variable(shape=(label_num,768),dtype=tf.float16,	
                                initializer=tf.truncated_normal_initializer(stddev=0.02),
                                name="output_weights")
        saved_variables.append(loss_weight)
        loss_bias = tf.get_variable(shape=(label_num,),dtype=tf.float16,
                                initializer=tf.zeros_initializer(),	
                                name="output_bias")
        saved_variables.append(loss_bias)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)	
        print("Save to :" + output_file)

def convert_compare_ipu_gpu(ckpt_a, ckpt_b):
    graph = tf.Graph()
    reader_a = pywrap_tensorflow.NewCheckpointReader(ckpt_a)
    reader_b = pywrap_tensorflow.NewCheckpointReader(ckpt_b)
    var_to_shape_map_a = reader_a.get_variable_to_shape_map()
    var_to_shape_map_b = reader_b.get_variable_to_shape_map()
    import pdb
    pdb.set_trace()
    with graph.as_default():
        sess = tf.Session()
        for tensor_name in var_to_shape_map_a:
            tensor_value_a = reader_a.get_tensor(tensor_name)
            tensor_value_b = reader_b.get_tensor(tensor_name)
            if tensor_value_a != tensor_value_b:
                print(tensor_name)
    
    print("finish compare!")


def convert_gc_ckpt_to_google(ckpt_file,
                              output_dir=None,
                              include_qkv_bias=False,
                              dtype=tf.float32):
    """ Convert GC bert checkpoint to Google original checkpoint
        1. combine `word_embeddings` if splitted
        2. rename scope `bert/encoder/layer_x/attention/projection/` to `bert/encoder/layer_x/attention/output/`
        3. add back attention_projection_bias.
        4. split `qkv_weight` to query,key,value, and add relative bias.
        5. rename `GroupNorm` to `LayerNorm`.
        6. add back dense layer before mlm loss.
    Args:
        ckpt_file: str, Google checkpoint.
        output_dir: str, Path to save converted GC checkpoint.
        include_qkv_bias: bool, are there bias weights in attention layer.
        dtype: tf.float32 or tf.float16, type of tensor in output ckpt file. Only will be used when load origin google checkpoint

    Returns:
        None
    """
    graph = tf.Graph()
    dir_name, ckpt_name = os.path.split(os.path.abspath(ckpt_file))
    if not output_dir:
        output_dir = os.path.join(dir_name, "google_ckpt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.Session()
        num_hidden_layers = 0
        optimizer_names = ["adam", "Momentum", "lamb"]  # optimizer weights
        word_embeddings = []
        new_variables = []
        keep_vardiables = []
        for tensor_name in var_to_shape_map:
            # logger.info(f"Load {tensor_name}......")
            # Filter the optimizer variables
            if filter_optimizer(tensor_name, optimizer_names):
                continue

            tensor_value = tf.cast(reader.get_tensor(tensor_name), dtype=dtype)

            if tensor_name == 'bert/encoder/layer_0/intermediate/dense/kernel' or tensor_name == 'bert/pooler/dense/kernel':
                print(tensor_name)
                print(tensor_value)
            # logger.info(f"Shape is {tensor_value.shape}")
            if "word_embeddings" in tensor_name:
                word_embeddings.append(tensor_name)
            elif "attention" in tensor_name:
                layer_idx = int(tensor_name.split("/")[2].split("_")[-1])
                num_hidden_layers = max(layer_idx, num_hidden_layers)
                # split query, key, value.
                if "qkv_weight" in tensor_name:
                    hidden_size = tensor_value.shape[1]//3
                    query = tensor_value[:, :hidden_size]
                    key = tensor_value[:, hidden_size:2*hidden_size]
                    value = tensor_value[:, 2*hidden_size:]

                    qw = tf.Variable(query, name=tensor_name.replace("qkv_weight", "query/kernel"))
                    kw = tf.Variable(key, name=tensor_name.replace("qkv_weight", "key/kernel"))
                    vw = tf.Variable(value, name=tensor_name.replace("qkv_weight", "value/kernel"))
                    new_variables.extend([qw, kw, vw])
                elif "qkv_bias" in tensor_name and include_qkv_bias:
                    hidden_size = tensor_value.shape[0]//3
                    query_bias = tensor_value[:hidden_size]
                    key_bias = tensor_value[hidden_size:2*hidden_size]
                    value_bias = tensor_value[2*hidden_size:]
                    qb = tf.Variable(query_bias, name=tensor_name.replace("qkv_bias", "query/bias"))
                    kb = tf.Variable(key_bias, name=tensor_name.replace("qkv_bias", "key/bias"))
                    vb = tf.Variable(value_bias, name=tensor_name.replace("qkv_bias", "value/bias"))
                    new_variables.extend([qb, kb, vb])
                # rename projection to output
                elif "projection" in tensor_name:
                    # logger.debug(f"Rename projection......")
                    new_name = tensor_name.replace("projection", "output")
                    if "GroupNorm" in tensor_name:
                        # logger.debug(f"Rename GroupNorm in attention ......")
                        new_name = new_name.replace("GroupNorm", "LayerNorm")

                    proj = tf.Variable(tensor_value, name=new_name)
                    new_variables.append(proj)
            # rename other GroupNorm
            elif "GroupNorm" in tensor_name:
                # logger.debug(f"Rename GroupNorm ......")
                gn = tf.Variable(tensor_value, name=tensor_name.replace("GroupNorm", "LayerNorm"))
                new_variables.append(gn)
            else:
                var = tf.get_variable(tensor_name, shape=tensor_value.shape, dtype=dtype)
                # var = tf.Variable(tensor_value, name=tensor_name)
                keep_vardiables.append(var)

        # Combine splitted embeddings
        word_embeddings = np.sort(word_embeddings)
        embeddings_vals = [reader.get_tensor(k) for k in word_embeddings]
        unit_embeddings = np.vstack(embeddings_vals)
        # logger.debug(f"Concated word_embeddings shape: {unit_embeddings.shape}")
        we = tf.Variable(
            unit_embeddings,
            dtype=dtype,
            shape=unit_embeddings.shape, 
            name="bert/embeddings/word_embeddings")
        new_variables.append(we)
        saved_variables = new_variables + keep_vardiables
        # logger.info("Finish concat word embeddings.")
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)
        # logger.info("Save to :" + output_file)

if __name__=='__main__':
    if args.convert:
        convert_ckpt_to_fp(args.ckpt_dir)
    if args.remove:
        remove_train_cache_parameters(args.ckpt_dir)
    if args.print:
        print_ckpt_tensor_name(args.ckpt_dir)
    if args.rename:
        rename_ckpt_tensor_name(args.ckpt_dir)
    if args.googletogc:
        convert_ipu_ckpt_to_gc(args.ckpt_dir,use_attention_bias=True,use_qkv_bias=True,label_num=2)
    if args.compare:
        convert_compare_ipu_gpu(args.ckpt_dir,args.ckpt_dir_b)
    if args.gctogoogle:
        convert_gc_ckpt_to_google(args.ckpt_dir,include_qkv_bias=True)
