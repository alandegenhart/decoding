��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8��
y
dense_3/kernelVarHandleOp*
shape:	�*
shared_namedense_3/kernel*
dtype0*
_output_shapes
: 
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes
:	�
p
dense_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: *
shape: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shared_name
Adam/decay*
dtype0*
_output_shapes
: *
shape: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
simple_rnn_3/kernelVarHandleOp*
_output_shapes
: *
shape:	�*$
shared_namesimple_rnn_3/kernel*
dtype0
|
'simple_rnn_3/kernel/Read/ReadVariableOpReadVariableOpsimple_rnn_3/kernel*
dtype0*
_output_shapes
:	�
�
simple_rnn_3/recurrent_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*.
shared_namesimple_rnn_3/recurrent_kernel
�
1simple_rnn_3/recurrent_kernel/Read/ReadVariableOpReadVariableOpsimple_rnn_3/recurrent_kernel*
dtype0* 
_output_shapes
:
��
{
simple_rnn_3/biasVarHandleOp*"
shared_namesimple_rnn_3/bias*
dtype0*
_output_shapes
: *
shape:�
t
%simple_rnn_3/bias/Read/ReadVariableOpReadVariableOpsimple_rnn_3/bias*
dtype0*
_output_shapes	
:�
�
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
shape:	�*&
shared_nameAdam/dense_3/kernel/m*
dtype0
�
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
dtype0*
_output_shapes
:	�
~
Adam/dense_3/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
dtype0*
_output_shapes
:
�
Adam/simple_rnn_3/kernel/mVarHandleOp*
_output_shapes
: *
shape:	�*+
shared_nameAdam/simple_rnn_3/kernel/m*
dtype0
�
.Adam/simple_rnn_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/simple_rnn_3/kernel/m*
dtype0*
_output_shapes
:	�
�
$Adam/simple_rnn_3/recurrent_kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*5
shared_name&$Adam/simple_rnn_3/recurrent_kernel/m
�
8Adam/simple_rnn_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/simple_rnn_3/recurrent_kernel/m*
dtype0* 
_output_shapes
:
��
�
Adam/simple_rnn_3/bias/mVarHandleOp*
shape:�*)
shared_nameAdam/simple_rnn_3/bias/m*
dtype0*
_output_shapes
: 
�
,Adam/simple_rnn_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/simple_rnn_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_3/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	�*&
shared_nameAdam/dense_3/kernel/v
�
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
dtype0*
_output_shapes
:	�
~
Adam/dense_3/bias/vVarHandleOp*$
shared_nameAdam/dense_3/bias/v*
dtype0*
_output_shapes
: *
shape:
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
dtype0*
_output_shapes
:
�
Adam/simple_rnn_3/kernel/vVarHandleOp*
shape:	�*+
shared_nameAdam/simple_rnn_3/kernel/v*
dtype0*
_output_shapes
: 
�
.Adam/simple_rnn_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/simple_rnn_3/kernel/v*
dtype0*
_output_shapes
:	�
�
$Adam/simple_rnn_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
shape:
��*5
shared_name&$Adam/simple_rnn_3/recurrent_kernel/v*
dtype0
�
8Adam/simple_rnn_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/simple_rnn_3/recurrent_kernel/v*
dtype0* 
_output_shapes
:
��
�
Adam/simple_rnn_3/bias/vVarHandleOp*)
shared_nameAdam/simple_rnn_3/bias/v*
dtype0*
_output_shapes
: *
shape:�
�
,Adam/simple_rnn_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/simple_rnn_3/bias/v*
dtype0*
_output_shapes	
:�

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
R

trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�
iter

beta_1

beta_2
	decay
learning_ratem:m;m< m=!m>v?v@vA vB!vC
#
0
 1
!2
3
4
#
0
 1
!2
3
4
 
�
"non_trainable_variables

#layers
trainable_variables
	variables
$metrics
regularization_losses
%layer_regularization_losses
 
 
 
 
�
&non_trainable_variables

'layers

trainable_variables
	variables
(metrics
regularization_losses
)layer_regularization_losses
~

kernel
 recurrent_kernel
!bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
 

0
 1
!2

0
 1
!2
 
�
.non_trainable_variables

/layers
trainable_variables
	variables
0metrics
regularization_losses
1layer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
2non_trainable_variables

3layers
trainable_variables
	variables
4metrics
regularization_losses
5layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsimple_rnn_3/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsimple_rnn_3/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsimple_rnn_3/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
 
 
 
 

0
 1
!2

0
 1
!2
 
�
6non_trainable_variables

7layers
*trainable_variables
+	variables
8metrics
,regularization_losses
9layer_regularization_losses
 

0
 
 
 
 
 
 
 
 
 
 
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/simple_rnn_3/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE$Adam/simple_rnn_3/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/simple_rnn_3/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/simple_rnn_3/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE$Adam/simple_rnn_3/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/simple_rnn_3/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
�
"serving_default_simple_rnn_3_inputPlaceholder*
dtype0*4
_output_shapes"
 :������������������*)
shape :������������������
�
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_3_inputsimple_rnn_3/kernelsimple_rnn_3/biassimple_rnn_3/recurrent_kerneldense_3/kerneldense_3/bias**
config_proto

CPU

GPU 2J 8*
Tin

2*4
_output_shapes"
 :������������������*,
_gradient_op_typePartitionedCall-43556*,
f'R%
#__inference_signature_wrapper_42551*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'simple_rnn_3/kernel/Read/ReadVariableOp1simple_rnn_3/recurrent_kernel/Read/ReadVariableOp%simple_rnn_3/bias/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp.Adam/simple_rnn_3/kernel/m/Read/ReadVariableOp8Adam/simple_rnn_3/recurrent_kernel/m/Read/ReadVariableOp,Adam/simple_rnn_3/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp.Adam/simple_rnn_3/kernel/v/Read/ReadVariableOp8Adam/simple_rnn_3/recurrent_kernel/v/Read/ReadVariableOp,Adam/simple_rnn_3/bias/v/Read/ReadVariableOpConst**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *!
Tin
2	*,
_gradient_op_typePartitionedCall-43598*'
f"R 
__inference__traced_save_43597*
Tout
2
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratesimple_rnn_3/kernelsimple_rnn_3/recurrent_kernelsimple_rnn_3/biasAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/simple_rnn_3/kernel/m$Adam/simple_rnn_3/recurrent_kernel/mAdam/simple_rnn_3/bias/mAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/simple_rnn_3/kernel/v$Adam/simple_rnn_3/recurrent_kernel/vAdam/simple_rnn_3/bias/v* 
Tin
2*
_output_shapes
: *,
_gradient_op_typePartitionedCall-43671**
f%R#
!__inference__traced_restore_43670*
Tout
2**
config_proto

CPU

GPU 2J 8��
�A
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43274
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0N
zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0Q
zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*4
_output_shapes"
 :������������������*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"�����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resourcebiasadd_readvariableop_resource matmul_1_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_43195*:
_output_shapes(
&: : : : :����������: : : : : *
T
2
*9
output_shapes(
&: : : : :����������: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_43194*
_num_original_outputs
K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0_
while/Identity_4Identitywhile:output:4*(
_output_shapes
:����������*
T0M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*(
_output_shapes
:����������e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*5
_output_shapes#
!:�������������������*
T0�
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :( $
"
_user_specified_name
inputs/0: : 
�
�
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41656

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*(
_output_shapes
:����������*
T0H
TanhTanhadd:z:0*
T0*(
_output_shapes
:�����������
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*(
_output_shapes
:����������*
T0�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:���������:����������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : :& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�A
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43133

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0N
zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
dtype0*
_output_shapes
: *
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
��*
dtype0v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*(
_output_shapes
:����������*
T0H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"�����   *
dtype0�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resourcebiasadd_readvariableop_resource matmul_1_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs
*
bodyR
while_body_43054*:
_output_shapes(
&: : : : :����������: : : : : *
T
2
*9
output_shapes(
&: : : : :����������: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_43053K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: _
while/Identity_4Identitywhile:output:4*
T0*(
_output_shapes
:����������M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*(
_output_shapes
:����������*
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*5
_output_shapes#
!:�������������������*
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs: : : 
�	
�
1__inference_simple_rnn_cell_3_layer_call_fn_43501

inputs
states_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41656*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*<
_output_shapes*
(:����������:����������*,
_gradient_op_typePartitionedCall-41680�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:�����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*(
_output_shapes
:����������*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:���������:����������:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: : : 
��
�
 __inference__wrapped_model_41605
simple_rnn_3_input<
8sequential_3_simple_rnn_3_matmul_readvariableop_resource=
9sequential_3_simple_rnn_3_biasadd_readvariableop_resource>
:sequential_3_simple_rnn_3_matmul_1_readvariableop_resource:
6sequential_3_dense_3_tensordot_readvariableop_resource8
4sequential_3_dense_3_biasadd_readvariableop_resource
identity��+sequential_3/dense_3/BiasAdd/ReadVariableOp�-sequential_3/dense_3/Tensordot/ReadVariableOp�0sequential_3/simple_rnn_3/BiasAdd/ReadVariableOp�/sequential_3/simple_rnn_3/MatMul/ReadVariableOp�1sequential_3/simple_rnn_3/MatMul_1/ReadVariableOp�sequential_3/simple_rnn_3/whilea
sequential_3/simple_rnn_3/ShapeShapesimple_rnn_3_input*
_output_shapes
:*
T0w
-sequential_3/simple_rnn_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/sequential_3/simple_rnn_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/sequential_3/simple_rnn_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'sequential_3/simple_rnn_3/strided_sliceStridedSlice(sequential_3/simple_rnn_3/Shape:output:06sequential_3/simple_rnn_3/strided_slice/stack:output:08sequential_3/simple_rnn_3/strided_slice/stack_1:output:08sequential_3/simple_rnn_3/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0h
%sequential_3/simple_rnn_3/zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: �
#sequential_3/simple_rnn_3/zeros/mulMul0sequential_3/simple_rnn_3/strided_slice:output:0.sequential_3/simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: i
&sequential_3/simple_rnn_3/zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: �
$sequential_3/simple_rnn_3/zeros/LessLess'sequential_3/simple_rnn_3/zeros/mul:z:0/sequential_3/simple_rnn_3/zeros/Less/y:output:0*
_output_shapes
: *
T0k
(sequential_3/simple_rnn_3/zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: �
&sequential_3/simple_rnn_3/zeros/packedPack0sequential_3/simple_rnn_3/strided_slice:output:01sequential_3/simple_rnn_3/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:j
%sequential_3/simple_rnn_3/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
sequential_3/simple_rnn_3/zerosFill/sequential_3/simple_rnn_3/zeros/packed:output:0.sequential_3/simple_rnn_3/zeros/Const:output:0*
T0*(
_output_shapes
:����������}
(sequential_3/simple_rnn_3/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
#sequential_3/simple_rnn_3/transpose	Transposesimple_rnn_3_input1sequential_3/simple_rnn_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������x
!sequential_3/simple_rnn_3/Shape_1Shape'sequential_3/simple_rnn_3/transpose:y:0*
T0*
_output_shapes
:y
/sequential_3/simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0{
1sequential_3/simple_rnn_3/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1sequential_3/simple_rnn_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
)sequential_3/simple_rnn_3/strided_slice_1StridedSlice*sequential_3/simple_rnn_3/Shape_1:output:08sequential_3/simple_rnn_3/strided_slice_1/stack:output:0:sequential_3/simple_rnn_3/strided_slice_1/stack_1:output:0:sequential_3/simple_rnn_3/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
5sequential_3/simple_rnn_3/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
'sequential_3/simple_rnn_3/TensorArrayV2TensorListReserve>sequential_3/simple_rnn_3/TensorArrayV2/element_shape:output:02sequential_3/simple_rnn_3/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
Osequential_3/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
Asequential_3/simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_3/simple_rnn_3/transpose:y:0Xsequential_3/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: y
/sequential_3/simple_rnn_3/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1sequential_3/simple_rnn_3/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1sequential_3/simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
)sequential_3/simple_rnn_3/strided_slice_2StridedSlice'sequential_3/simple_rnn_3/transpose:y:08sequential_3/simple_rnn_3/strided_slice_2/stack:output:0:sequential_3/simple_rnn_3/strided_slice_2/stack_1:output:0:sequential_3/simple_rnn_3/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
T0*
Index0�
/sequential_3/simple_rnn_3/MatMul/ReadVariableOpReadVariableOp8sequential_3_simple_rnn_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	�*
dtype0�
 sequential_3/simple_rnn_3/MatMulMatMul2sequential_3/simple_rnn_3/strided_slice_2:output:07sequential_3/simple_rnn_3/MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
0sequential_3/simple_rnn_3/BiasAdd/ReadVariableOpReadVariableOp9sequential_3_simple_rnn_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:�*
dtype0�
!sequential_3/simple_rnn_3/BiasAddBiasAdd*sequential_3/simple_rnn_3/MatMul:product:08sequential_3/simple_rnn_3/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
1sequential_3/simple_rnn_3/MatMul_1/ReadVariableOpReadVariableOp:sequential_3_simple_rnn_3_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
"sequential_3/simple_rnn_3/MatMul_1MatMul(sequential_3/simple_rnn_3/zeros:output:09sequential_3/simple_rnn_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_3/simple_rnn_3/addAddV2*sequential_3/simple_rnn_3/BiasAdd:output:0,sequential_3/simple_rnn_3/MatMul_1:product:0*
T0*(
_output_shapes
:����������|
sequential_3/simple_rnn_3/TanhTanh!sequential_3/simple_rnn_3/add:z:0*
T0*(
_output_shapes
:�����������
7sequential_3/simple_rnn_3/TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
)sequential_3/simple_rnn_3/TensorArrayV2_1TensorListReserve@sequential_3/simple_rnn_3/TensorArrayV2_1/element_shape:output:02sequential_3/simple_rnn_3/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: `
sequential_3/simple_rnn_3/timeConst*
value	B : *
dtype0*
_output_shapes
: }
2sequential_3/simple_rnn_3/while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: n
,sequential_3/simple_rnn_3/while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : �
sequential_3/simple_rnn_3/whileWhile5sequential_3/simple_rnn_3/while/loop_counter:output:0;sequential_3/simple_rnn_3/while/maximum_iterations:output:0'sequential_3/simple_rnn_3/time:output:02sequential_3/simple_rnn_3/TensorArrayV2_1:handle:0(sequential_3/simple_rnn_3/zeros:output:02sequential_3/simple_rnn_3/strided_slice_1:output:0Qsequential_3/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_3_simple_rnn_3_matmul_readvariableop_resource9sequential_3_simple_rnn_3_biasadd_readvariableop_resource:sequential_3_simple_rnn_3_matmul_1_readvariableop_resource1^sequential_3/simple_rnn_3/BiasAdd/ReadVariableOp0^sequential_3/simple_rnn_3/MatMul/ReadVariableOp2^sequential_3/simple_rnn_3/MatMul_1/ReadVariableOp*
_num_original_outputs
*6
body.R,
*sequential_3_simple_rnn_3_while_body_41496*:
_output_shapes(
&: : : : :����������: : : : : *
T
2
*9
output_shapes(
&: : : : :����������: : : : : *
_lower_using_switch_merge(*
parallel_iterations *6
cond.R,
*sequential_3_simple_rnn_3_while_cond_41495
(sequential_3/simple_rnn_3/while/IdentityIdentity(sequential_3/simple_rnn_3/while:output:0*
T0*
_output_shapes
: �
*sequential_3/simple_rnn_3/while/Identity_1Identity(sequential_3/simple_rnn_3/while:output:1*
_output_shapes
: *
T0�
*sequential_3/simple_rnn_3/while/Identity_2Identity(sequential_3/simple_rnn_3/while:output:2*
T0*
_output_shapes
: �
*sequential_3/simple_rnn_3/while/Identity_3Identity(sequential_3/simple_rnn_3/while:output:3*
_output_shapes
: *
T0�
*sequential_3/simple_rnn_3/while/Identity_4Identity(sequential_3/simple_rnn_3/while:output:4*
T0*(
_output_shapes
:�����������
*sequential_3/simple_rnn_3/while/Identity_5Identity(sequential_3/simple_rnn_3/while:output:5*
T0*
_output_shapes
: �
*sequential_3/simple_rnn_3/while/Identity_6Identity(sequential_3/simple_rnn_3/while:output:6*
T0*
_output_shapes
: �
*sequential_3/simple_rnn_3/while/Identity_7Identity(sequential_3/simple_rnn_3/while:output:7*
T0*
_output_shapes
: �
*sequential_3/simple_rnn_3/while/Identity_8Identity(sequential_3/simple_rnn_3/while:output:8*
T0*
_output_shapes
: �
*sequential_3/simple_rnn_3/while/Identity_9Identity(sequential_3/simple_rnn_3/while:output:9*
T0*
_output_shapes
: �
Jsequential_3/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
<sequential_3/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStack3sequential_3/simple_rnn_3/while/Identity_3:output:0Ssequential_3/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:��������������������
/sequential_3/simple_rnn_3/strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:{
1sequential_3/simple_rnn_3/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:{
1sequential_3/simple_rnn_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
)sequential_3/simple_rnn_3/strided_slice_3StridedSliceEsequential_3/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:08sequential_3/simple_rnn_3/strided_slice_3/stack:output:0:sequential_3/simple_rnn_3/strided_slice_3/stack_1:output:0:sequential_3/simple_rnn_3/strided_slice_3/stack_2:output:0*
shrink_axis_mask*(
_output_shapes
:����������*
Index0*
T0
*sequential_3/simple_rnn_3/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
%sequential_3/simple_rnn_3/transpose_1	TransposeEsequential_3/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:03sequential_3/simple_rnn_3/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
-sequential_3/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�m
#sequential_3/dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:t
#sequential_3/dense_3/Tensordot/freeConst*
_output_shapes
:*
valueB"       *
dtype0}
$sequential_3/dense_3/Tensordot/ShapeShape)sequential_3/simple_rnn_3/transpose_1:y:0*
T0*
_output_shapes
:n
,sequential_3/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
'sequential_3/dense_3/Tensordot/GatherV2GatherV2-sequential_3/dense_3/Tensordot/Shape:output:0,sequential_3/dense_3/Tensordot/free:output:05sequential_3/dense_3/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0p
.sequential_3/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0�
)sequential_3/dense_3/Tensordot/GatherV2_1GatherV2-sequential_3/dense_3/Tensordot/Shape:output:0,sequential_3/dense_3/Tensordot/axes:output:07sequential_3/dense_3/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0n
$sequential_3/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
#sequential_3/dense_3/Tensordot/ProdProd0sequential_3/dense_3/Tensordot/GatherV2:output:0-sequential_3/dense_3/Tensordot/Const:output:0*
_output_shapes
: *
T0p
&sequential_3/dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
%sequential_3/dense_3/Tensordot/Prod_1Prod2sequential_3/dense_3/Tensordot/GatherV2_1:output:0/sequential_3/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_3/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0�
%sequential_3/dense_3/Tensordot/concatConcatV2,sequential_3/dense_3/Tensordot/free:output:0,sequential_3/dense_3/Tensordot/axes:output:03sequential_3/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
$sequential_3/dense_3/Tensordot/stackPack,sequential_3/dense_3/Tensordot/Prod:output:0.sequential_3/dense_3/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
(sequential_3/dense_3/Tensordot/transpose	Transpose)sequential_3/simple_rnn_3/transpose_1:y:0.sequential_3/dense_3/Tensordot/concat:output:0*5
_output_shapes#
!:�������������������*
T0�
&sequential_3/dense_3/Tensordot/ReshapeReshape,sequential_3/dense_3/Tensordot/transpose:y:0-sequential_3/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/sequential_3/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
*sequential_3/dense_3/Tensordot/transpose_1	Transpose5sequential_3/dense_3/Tensordot/ReadVariableOp:value:08sequential_3/dense_3/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	�
.sequential_3/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�      *
dtype0*
_output_shapes
:�
(sequential_3/dense_3/Tensordot/Reshape_1Reshape.sequential_3/dense_3/Tensordot/transpose_1:y:07sequential_3/dense_3/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	��
%sequential_3/dense_3/Tensordot/MatMulMatMul/sequential_3/dense_3/Tensordot/Reshape:output:01sequential_3/dense_3/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������p
&sequential_3/dense_3/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:n
,sequential_3/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0�
'sequential_3/dense_3/Tensordot/concat_1ConcatV20sequential_3/dense_3/Tensordot/GatherV2:output:0/sequential_3/dense_3/Tensordot/Const_2:output:05sequential_3/dense_3/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential_3/dense_3/TensordotReshape/sequential_3/dense_3/Tensordot/MatMul:product:00sequential_3/dense_3/Tensordot/concat_1:output:0*4
_output_shapes"
 :������������������*
T0�
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
sequential_3/dense_3/BiasAddBiasAdd'sequential_3/dense_3/Tensordot:output:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*4
_output_shapes"
 :������������������*
T0�
IdentityIdentity%sequential_3/dense_3/BiasAdd:output:0,^sequential_3/dense_3/BiasAdd/ReadVariableOp.^sequential_3/dense_3/Tensordot/ReadVariableOp1^sequential_3/simple_rnn_3/BiasAdd/ReadVariableOp0^sequential_3/simple_rnn_3/MatMul/ReadVariableOp2^sequential_3/simple_rnn_3/MatMul_1/ReadVariableOp ^sequential_3/simple_rnn_3/while*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2B
sequential_3/simple_rnn_3/whilesequential_3/simple_rnn_3/while2d
0sequential_3/simple_rnn_3/BiasAdd/ReadVariableOp0sequential_3/simple_rnn_3/BiasAdd/ReadVariableOp2f
1sequential_3/simple_rnn_3/MatMul_1/ReadVariableOp1sequential_3/simple_rnn_3/MatMul_1/ReadVariableOp2b
/sequential_3/simple_rnn_3/MatMul/ReadVariableOp/sequential_3/simple_rnn_3/MatMul/ReadVariableOp2^
-sequential_3/dense_3/Tensordot/ReadVariableOp-sequential_3/dense_3/Tensordot/ReadVariableOp:2 .
,
_user_specified_namesimple_rnn_3_input: : : : : 
�!
�
while_body_43054
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
value	B :*
dtype0N
add_1AddV2placeholderadd_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_2AddV2while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*(
_output_shapes
:����������*
T0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0*?
_input_shapes.
,: : : : :����������: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 
�A
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42400

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskN
zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: m
zerosFillzeros/packed:output:0zeros/Const:output:0*(
_output_shapes
:����������*
T0c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :������������������*
T0D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"����   *
dtype0�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0_
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0e
addAddV2BiasAdd:output:0MatMul_1:product:0*(
_output_shapes
:����������*
T0H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0n
TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resourcebiasadd_readvariableop_resource matmul_1_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*:
_output_shapes(
&: : : : :����������: : : : : *9
output_shapes(
&: : : : :����������: : : : : *
T
2
*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_42320*
_num_original_outputs
*
bodyR
while_body_42321K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0_
while/Identity_4Identitywhile:output:4*
T0*(
_output_shapes
:����������M
while/Identity_5Identitywhile:output:5*
_output_shapes
: *
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*(
_output_shapes
:����������*
Index0*
T0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : : :& "
 
_user_specified_nameinputs
�
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_42526

inputs/
+simple_rnn_3_statefulpartitionedcall_args_1/
+simple_rnn_3_statefulpartitionedcall_args_2/
+simple_rnn_3_statefulpartitionedcall_args_3*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_3/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCallinputs+simple_rnn_3_statefulpartitionedcall_args_1+simple_rnn_3_statefulpartitionedcall_args_2+simple_rnn_3_statefulpartitionedcall_args_3*
Tout
2**
config_proto

CPU

GPU 2J 8*5
_output_shapes#
!:�������������������*
Tin
2*,
_gradient_op_typePartitionedCall-42412*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42400�
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*4
_output_shapes"
 :������������������*,
_gradient_op_typePartitionedCall-42463*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42457*
Tout
2�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : 
�0
�	
__inference__traced_save_43597
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_simple_rnn_3_kernel_read_readvariableop<
8savev2_simple_rnn_3_recurrent_kernel_read_readvariableop0
,savev2_simple_rnn_3_bias_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop9
5savev2_adam_simple_rnn_3_kernel_m_read_readvariableopC
?savev2_adam_simple_rnn_3_recurrent_kernel_m_read_readvariableop7
3savev2_adam_simple_rnn_3_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop9
5savev2_adam_simple_rnn_3_kernel_v_read_readvariableopC
?savev2_adam_simple_rnn_3_recurrent_kernel_v_read_readvariableop7
3savev2_adam_simple_rnn_3_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_58b786a1989642929c1bc88e76a0387e/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*�

value�
B�
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
SaveV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_simple_rnn_3_kernel_read_readvariableop8savev2_simple_rnn_3_recurrent_kernel_read_readvariableop,savev2_simple_rnn_3_bias_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop5savev2_adam_simple_rnn_3_kernel_m_read_readvariableop?savev2_adam_simple_rnn_3_recurrent_kernel_m_read_readvariableop3savev2_adam_simple_rnn_3_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop5savev2_adam_simple_rnn_3_kernel_v_read_readvariableop?savev2_adam_simple_rnn_3_recurrent_kernel_v_read_readvariableop3savev2_adam_simple_rnn_3_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *"
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:: : : : : :	�:
��:�:	�::	�:
��:�:	�::	�:
��:�: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2: : : : : : : : :	 :
 : : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix
�
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_42488
simple_rnn_3_input/
+simple_rnn_3_statefulpartitionedcall_args_1/
+simple_rnn_3_statefulpartitionedcall_args_2/
+simple_rnn_3_statefulpartitionedcall_args_3*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_3/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_3_input+simple_rnn_3_statefulpartitionedcall_args_1+simple_rnn_3_statefulpartitionedcall_args_2+simple_rnn_3_statefulpartitionedcall_args_3**
config_proto

CPU

GPU 2J 8*
Tin
2*5
_output_shapes#
!:�������������������*,
_gradient_op_typePartitionedCall-42412*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42400*
Tout
2�
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*4
_output_shapes"
 :������������������*
Tin
2*,
_gradient_op_typePartitionedCall-42463*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42457�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:2 .
,
_user_specified_namesimple_rnn_3_input: : : : : 
�A
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42275

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: N
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value
B :�_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: m
zerosFillzeros/packed:output:0zeros/Const:output:0*(
_output_shapes
:����������*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :������������������*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"�����   *
dtype0�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resourcebiasadd_readvariableop_resource matmul_1_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*9
output_shapes(
&: : : : :����������: : : : : *
T
2
*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_42195*
_num_original_outputs
*
bodyR
while_body_42196*:
_output_shapes(
&: : : : :����������: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: _
while/Identity_4Identitywhile:output:4*
T0*(
_output_shapes
:����������M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*(
_output_shapes
:����������*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*5
_output_shapes#
!:�������������������*
T0�
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile: :& "
 
_user_specified_nameinputs: : 
�
�
while_cond_43319
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : ::::  : : : : : : : : :	 
�
�
while_cond_42320
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : :::: : : : : : :	 :  : : 
�!
�
while_body_43195
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*
T0*(
_output_shapes
:�����������
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_1AddV2placeholderadd_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_2AddV2while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*?
_input_shapes.
,: : : : :����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : :	 :  : : : : : 
�
�
while_body_42067
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_2_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2 statefulpartitionedcall_args_2_0 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0**
config_proto

CPU

GPU 2J 8*<
_output_shapes*
(:����������:����������*
Tin	
2*,
_gradient_op_typePartitionedCall-41694*U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41675*
Tout
2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
_output_shapes
: *
T0I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
_output_shapes
: *
T0Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
_output_shapes
: *
T0�

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0"B
statefulpartitionedcall_args_2 statefulpartitionedcall_args_2_0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes.
,: : : : :����������: : :::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : :	 :  : : 
�$
�
*sequential_3_simple_rnn_3_while_body_414960
,sequential_3_simple_rnn_3_while_loop_counter6
2sequential_3_simple_rnn_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2/
+sequential_3_simple_rnn_3_strided_slice_1_0k
gtensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4-
)sequential_3_simple_rnn_3_strided_slice_1i
etensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemgtensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_1AddV2placeholderadd_1/y:output:0*
_output_shapes
: *
T0I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: o
add_2AddV2,sequential_3_simple_rnn_3_while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identity2sequential_3_simple_rnn_3_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*(
_output_shapes
:����������*
T0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"X
)sequential_3_simple_rnn_3_strided_slice_1+sequential_3_simple_rnn_3_strided_slice_1_0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"�
etensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorgtensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*?
_input_shapes.
,: : : : :����������: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : : : : : :	 :  : : 
�
�
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_43473

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*(
_output_shapes
:����������*
T0�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:���������:����������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: 
�
�
#__inference_signature_wrapper_42551
simple_rnn_3_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_3_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*)
f$R"
 __inference__wrapped_model_41605*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin

2*4
_output_shapes"
 :������������������*,
_gradient_op_typePartitionedCall-42543�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::22
StatefulPartitionedCallStatefulPartitionedCall:2 .
,
_user_specified_namesimple_rnn_3_input: : : : : 
�
�
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_43490

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*
T0*(
_output_shapes
:�����������
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*(
_output_shapes
:����������*
T0�

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0"
identityIdentity:output:0*F
_input_shapes5
3:���������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: : 
�
�
,__inference_simple_rnn_3_layer_call_fn_43407
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-42017*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42016*
Tout
2**
config_proto

CPU

GPU 2J 8*5
_output_shapes#
!:�������������������*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
�#
�
simple_rnn_3_while_body_42599#
simple_rnn_3_while_loop_counter)
%simple_rnn_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2"
simple_rnn_3_strided_slice_1_0^
Ztensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4 
simple_rnn_3_strided_slice_1\
Xtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	�*
dtype0�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
value	B :*
dtype0N
add_1AddV2placeholderadd_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: b
add_2AddV2simple_rnn_3_while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identity%simple_rnn_3_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"�
Xtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorZtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0">
simple_rnn_3_strided_slice_1simple_rnn_3_strided_slice_1_0"!

identity_1Identity_1:output:0*?
_input_shapes.
,: : : : :����������: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : :	 :  : : : 
�
�
*sequential_3_simple_rnn_3_while_cond_414950
,sequential_3_simple_rnn_3_while_loop_counter6
2sequential_3_simple_rnn_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_22
.less_sequential_3_simple_rnn_3_strided_slice_1E
Asequential_3_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
j
LessLessplaceholder.less_sequential_3_simple_rnn_3_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : ::::  : : : : : : : : :	 
�!
�
while_body_42196
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*(
_output_shapes
:����������*
T0H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_1AddV2placeholderadd_1/y:output:0*
_output_shapes
: *
T0I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_2AddV2while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*?
_input_shapes.
,: : : : :����������: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 
�Q
�
!__inference__traced_restore_43670
file_prefix#
assignvariableop_dense_3_kernel#
assignvariableop_1_dense_3_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate*
&assignvariableop_7_simple_rnn_3_kernel4
0assignvariableop_8_simple_rnn_3_recurrent_kernel(
$assignvariableop_9_simple_rnn_3_bias-
)assignvariableop_10_adam_dense_3_kernel_m+
'assignvariableop_11_adam_dense_3_bias_m2
.assignvariableop_12_adam_simple_rnn_3_kernel_m<
8assignvariableop_13_adam_simple_rnn_3_recurrent_kernel_m0
,assignvariableop_14_adam_simple_rnn_3_bias_m-
)assignvariableop_15_adam_dense_3_kernel_v+
'assignvariableop_16_adam_dense_3_bias_v2
.assignvariableop_17_adam_simple_rnn_3_kernel_v<
8assignvariableop_18_adam_simple_rnn_3_recurrent_kernel_v0
,assignvariableop_19_adam_simple_rnn_3_bias_v
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�

RestoreV2/tensor_namesConst"/device:CPU:0*�

value�
B�
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
RestoreV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:{
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0	|
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
dtype0	*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:~
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0~
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:}
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_simple_rnn_3_kernelIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_simple_rnn_3_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_simple_rnn_3_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_adam_dense_3_kernel_mIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_dense_3_bias_mIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_adam_simple_rnn_3_kernel_mIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_simple_rnn_3_recurrent_kernel_mIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_simple_rnn_3_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_3_kernel_vIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_3_bias_vIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_simple_rnn_3_kernel_vIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0�
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_simple_rnn_3_recurrent_kernel_vIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_simple_rnn_3_bias_vIdentity_19:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0�
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : : : : :	 :
 : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix
�A
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43008

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0N
zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: m
zerosFillzeros/packed:output:0zeros/Const:output:0*(
_output_shapes
:����������*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :������������������*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
���������*
dtype0�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*(
_output_shapes
:����������*
T0H
TanhTanhadd:z:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
���������T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resourcebiasadd_readvariableop_resource matmul_1_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_42929*:
_output_shapes(
&: : : : :����������: : : : : *
T
2
*9
output_shapes(
&: : : : :����������: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_42928*
_num_original_outputs
K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0_
while/Identity_4Identitywhile:output:4*(
_output_shapes
:����������*
T0M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*(
_output_shapes
:����������*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : 
�	
�
,__inference_sequential_3_layer_call_fn_42535
simple_rnn_3_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_3_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-42527*P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_42526*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin

2*4
_output_shapes"
 :�������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*4
_output_shapes"
 :������������������*
T0"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::22
StatefulPartitionedCallStatefulPartitionedCall: : :2 .
,
_user_specified_namesimple_rnn_3_input: : : 
�
�
,__inference_sequential_3_layer_call_fn_42883

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-42527*P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_42526*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin

2*4
_output_shapes"
 :�������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : 
�#
�
simple_rnn_3_while_body_42754#
simple_rnn_3_while_loop_counter)
%simple_rnn_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2"
simple_rnn_3_strided_slice_1_0^
Ztensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4 
simple_rnn_3_strided_slice_1\
Xtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemZtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_1AddV2placeholderadd_1/y:output:0*
_output_shapes
: *
T0I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: b
add_2AddV2simple_rnn_3_while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identity%simple_rnn_3_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0">
simple_rnn_3_strided_slice_1simple_rnn_3_strided_slice_1_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"�
Xtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorZtensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :����������: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : :	 :  : : : : : : 
�
�
while_cond_43053
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : ::::  : : : : : : : : :	 
�
�
while_cond_43194
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : ::::  : : : : : : : : :	 
�
�
while_cond_42928
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : :::: : : : : : : :	 :  : 
�s
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_42863

inputs/
+simple_rnn_3_matmul_readvariableop_resource0
,simple_rnn_3_biasadd_readvariableop_resource1
-simple_rnn_3_matmul_1_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�#simple_rnn_3/BiasAdd/ReadVariableOp�"simple_rnn_3/MatMul/ReadVariableOp�$simple_rnn_3/MatMul_1/ReadVariableOp�simple_rnn_3/whileH
simple_rnn_3/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:l
"simple_rnn_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:l
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0[
simple_rnn_3/zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: �
simple_rnn_3/zeros/mulMul#simple_rnn_3/strided_slice:output:0!simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: \
simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
value
B :�*
dtype0�
simple_rnn_3/zeros/LessLesssimple_rnn_3/zeros/mul:z:0"simple_rnn_3/zeros/Less/y:output:0*
_output_shapes
: *
T0^
simple_rnn_3/zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: �
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:]
simple_rnn_3/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: �
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*(
_output_shapes
:����������p
simple_rnn_3/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
simple_rnn_3/transpose	Transposeinputs$simple_rnn_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������^
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0n
$simple_rnn_3/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: s
(simple_rnn_3/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0l
"simple_rnn_3/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:n
$simple_rnn_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_mask�
"simple_rnn_3/MatMul/ReadVariableOpReadVariableOp+simple_rnn_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
simple_rnn_3/MatMulMatMul%simple_rnn_3/strided_slice_2:output:0*simple_rnn_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#simple_rnn_3/BiasAdd/ReadVariableOpReadVariableOp,simple_rnn_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
simple_rnn_3/BiasAddBiasAddsimple_rnn_3/MatMul:product:0+simple_rnn_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$simple_rnn_3/MatMul_1/ReadVariableOpReadVariableOp-simple_rnn_3_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
simple_rnn_3/MatMul_1MatMulsimple_rnn_3/zeros:output:0,simple_rnn_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
simple_rnn_3/addAddV2simple_rnn_3/BiasAdd:output:0simple_rnn_3/MatMul_1:product:0*(
_output_shapes
:����������*
T0b
simple_rnn_3/TanhTanhsimple_rnn_3/add:z:0*
T0*(
_output_shapes
:����������{
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: S
simple_rnn_3/timeConst*
value	B : *
dtype0*
_output_shapes
: p
%simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
valueB :
���������*
dtype0a
simple_rnn_3/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0+simple_rnn_3_matmul_readvariableop_resource,simple_rnn_3_biasadd_readvariableop_resource-simple_rnn_3_matmul_1_readvariableop_resource$^simple_rnn_3/BiasAdd/ReadVariableOp#^simple_rnn_3/MatMul/ReadVariableOp%^simple_rnn_3/MatMul_1/ReadVariableOp*
parallel_iterations *)
cond!R
simple_rnn_3_while_cond_42753*
_num_original_outputs
*)
body!R
simple_rnn_3_while_body_42754*:
_output_shapes(
&: : : : :����������: : : : : *9
output_shapes(
&: : : : :����������: : : : : *
T
2
*
_lower_using_switch_merge(e
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while:output:0*
_output_shapes
: *
T0g
simple_rnn_3/while/Identity_1Identitysimple_rnn_3/while:output:1*
_output_shapes
: *
T0g
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while:output:2*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_3Identitysimple_rnn_3/while:output:3*
T0*
_output_shapes
: y
simple_rnn_3/while/Identity_4Identitysimple_rnn_3/while:output:4*
T0*(
_output_shapes
:����������g
simple_rnn_3/while/Identity_5Identitysimple_rnn_3/while:output:5*
_output_shapes
: *
T0g
simple_rnn_3/while/Identity_6Identitysimple_rnn_3/while:output:6*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_7Identitysimple_rnn_3/while:output:7*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_8Identitysimple_rnn_3/while:output:8*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_9Identitysimple_rnn_3/while:output:9*
_output_shapes
: *
T0�
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStack&simple_rnn_3/while/Identity_3:output:0Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������u
"simple_rnn_3/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������n
$simple_rnn_3/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
shrink_axis_mask*(
_output_shapes
:����������*
Index0*
T0r
simple_rnn_3/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	�*
dtype0`
dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:g
dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:c
dense_3/Tensordot/ShapeShapesimple_rnn_3/transpose_1:y:0*
_output_shapes
:*
T0a
dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0c
!dense_3/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0a
dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
_output_shapes
: *
T0_
dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0�
dense_3/Tensordot/transpose	Transposesimple_rnn_3/transpose_1:y:0!dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0s
"dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
dense_3/Tensordot/transpose_1	Transpose(dense_3/Tensordot/ReadVariableOp:value:0+dense_3/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	�r
!dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�      *
dtype0*
_output_shapes
:�
dense_3/Tensordot/Reshape_1Reshape!dense_3/Tensordot/transpose_1:y:0*dense_3/Tensordot/Reshape_1/shape:output:0*
_output_shapes
:	�*
T0�
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0$dense_3/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������c
dense_3/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:a
dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :�������������������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp$^simple_rnn_3/BiasAdd/ReadVariableOp#^simple_rnn_3/MatMul/ReadVariableOp%^simple_rnn_3/MatMul_1/ReadVariableOp^simple_rnn_3/while*4
_output_shapes"
 :������������������*
T0"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::2H
"simple_rnn_3/MatMul/ReadVariableOp"simple_rnn_3/MatMul/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while2J
#simple_rnn_3/BiasAdd/ReadVariableOp#simple_rnn_3/BiasAdd/ReadVariableOp2L
$simple_rnn_3/MatMul_1/ReadVariableOp$simple_rnn_3/MatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : 
�
�
,__inference_simple_rnn_3_layer_call_fn_43149

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*5
_output_shapes#
!:�������������������*,
_gradient_op_typePartitionedCall-42412*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42400*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs
�
�
simple_rnn_3_while_cond_42598#
simple_rnn_3_while_loop_counter)
%simple_rnn_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2%
!less_simple_rnn_3_strided_slice_18
4simple_rnn_3_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
]
LessLessplaceholder!less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : :::: : : : : :	 :  : : : 
�s
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_42708

inputs/
+simple_rnn_3_matmul_readvariableop_resource0
,simple_rnn_3_biasadd_readvariableop_resource1
-simple_rnn_3_matmul_1_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�#simple_rnn_3/BiasAdd/ReadVariableOp�"simple_rnn_3/MatMul/ReadVariableOp�$simple_rnn_3/MatMul_1/ReadVariableOp�simple_rnn_3/whileH
simple_rnn_3/ShapeShapeinputs*
_output_shapes
:*
T0j
 simple_rnn_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:l
"simple_rnn_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:l
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0[
simple_rnn_3/zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: �
simple_rnn_3/zeros/mulMul#simple_rnn_3/strided_slice:output:0!simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: \
simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
value
B :�*
dtype0�
simple_rnn_3/zeros/LessLesssimple_rnn_3/zeros/mul:z:0"simple_rnn_3/zeros/Less/y:output:0*
T0*
_output_shapes
: ^
simple_rnn_3/zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: �
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
_output_shapes
:*
T0*
N]
simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0�
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*(
_output_shapes
:����������p
simple_rnn_3/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
simple_rnn_3/transpose	Transposeinputs$simple_rnn_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������^
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_3/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: s
(simple_rnn_3/TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: l
"simple_rnn_3/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:����������
"simple_rnn_3/MatMul/ReadVariableOpReadVariableOp+simple_rnn_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
simple_rnn_3/MatMulMatMul%simple_rnn_3/strided_slice_2:output:0*simple_rnn_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#simple_rnn_3/BiasAdd/ReadVariableOpReadVariableOp,simple_rnn_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:�*
dtype0�
simple_rnn_3/BiasAddBiasAddsimple_rnn_3/MatMul:product:0+simple_rnn_3/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
$simple_rnn_3/MatMul_1/ReadVariableOpReadVariableOp-simple_rnn_3_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
simple_rnn_3/MatMul_1MatMulsimple_rnn_3/zeros:output:0,simple_rnn_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
simple_rnn_3/addAddV2simple_rnn_3/BiasAdd:output:0simple_rnn_3/MatMul_1:product:0*(
_output_shapes
:����������*
T0b
simple_rnn_3/TanhTanhsimple_rnn_3/add:z:0*
T0*(
_output_shapes
:����������{
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: S
simple_rnn_3/timeConst*
dtype0*
_output_shapes
: *
value	B : p
%simple_rnn_3/while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: a
simple_rnn_3/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0+simple_rnn_3_matmul_readvariableop_resource,simple_rnn_3_biasadd_readvariableop_resource-simple_rnn_3_matmul_1_readvariableop_resource$^simple_rnn_3/BiasAdd/ReadVariableOp#^simple_rnn_3/MatMul/ReadVariableOp%^simple_rnn_3/MatMul_1/ReadVariableOp*
T
2
*9
output_shapes(
&: : : : :����������: : : : : *
_lower_using_switch_merge(*
parallel_iterations *)
cond!R
simple_rnn_3_while_cond_42598*
_num_original_outputs
*)
body!R
simple_rnn_3_while_body_42599*:
_output_shapes(
&: : : : :����������: : : : : e
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while:output:0*
_output_shapes
: *
T0g
simple_rnn_3/while/Identity_1Identitysimple_rnn_3/while:output:1*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while:output:2*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_3Identitysimple_rnn_3/while:output:3*
_output_shapes
: *
T0y
simple_rnn_3/while/Identity_4Identitysimple_rnn_3/while:output:4*(
_output_shapes
:����������*
T0g
simple_rnn_3/while/Identity_5Identitysimple_rnn_3/while:output:5*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_6Identitysimple_rnn_3/while:output:6*
_output_shapes
: *
T0g
simple_rnn_3/while/Identity_7Identitysimple_rnn_3/while:output:7*
_output_shapes
: *
T0g
simple_rnn_3/while/Identity_8Identitysimple_rnn_3/while:output:8*
T0*
_output_shapes
: g
simple_rnn_3/while/Identity_9Identitysimple_rnn_3/while:output:9*
T0*
_output_shapes
: �
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStack&simple_rnn_3/while/Identity_3:output:0Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������u
"simple_rnn_3/strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:n
$simple_rnn_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*(
_output_shapes
:����������r
simple_rnn_3/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*5
_output_shapes#
!:�������������������*
T0�
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	�*
dtype0`
dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:g
dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:c
dense_3/Tensordot/ShapeShapesimple_rnn_3/transpose_1:y:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
_output_shapes
: *
T0c
dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
dense_3/Tensordot/transpose	Transposesimple_rnn_3/transpose_1:y:0!dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������s
"dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
dense_3/Tensordot/transpose_1	Transpose(dense_3/Tensordot/ReadVariableOp:value:0+dense_3/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	�r
!dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�      *
dtype0*
_output_shapes
:�
dense_3/Tensordot/Reshape_1Reshape!dense_3/Tensordot/transpose_1:y:0*dense_3/Tensordot/Reshape_1/shape:output:0*
_output_shapes
:	�*
T0�
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0$dense_3/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������c
dense_3/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:a
dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
_output_shapes
:*
T0*
N�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :�������������������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp$^simple_rnn_3/BiasAdd/ReadVariableOp#^simple_rnn_3/MatMul/ReadVariableOp%^simple_rnn_3/MatMul_1/ReadVariableOp^simple_rnn_3/while*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while2J
#simple_rnn_3/BiasAdd/ReadVariableOp#simple_rnn_3/BiasAdd/ReadVariableOp2L
$simple_rnn_3/MatMul_1/ReadVariableOp$simple_rnn_3/MatMul_1/ReadVariableOp2H
"simple_rnn_3/MatMul/ReadVariableOp"simple_rnn_3/MatMul/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : 
�!
�
while_body_43320
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0e
addAddV2BiasAdd:output:0MatMul_1:product:0*(
_output_shapes
:����������*
T0H
TanhTanhadd:z:0*
T0*(
_output_shapes
:�����������
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
value	B :*
dtype0N
add_1AddV2placeholderadd_1/y:output:0*
_output_shapes
: *
T0I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_2AddV2while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0*?
_input_shapes.
,: : : : :����������: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 
�
�
simple_rnn_3_while_cond_42753#
simple_rnn_3_while_loop_counter)
%simple_rnn_3_while_maximum_iterations
placeholder
placeholder_1
placeholder_2%
!less_simple_rnn_3_strided_slice_18
4simple_rnn_3_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
]
LessLessplaceholder!less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : ::::  : : : : : : : : :	 
�
�
while_body_41943
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_2_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2 statefulpartitionedcall_args_2_0 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0*<
_output_shapes*
(:����������:����������*
Tin	
2*,
_gradient_op_typePartitionedCall-41680*U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41656*
Tout
2**
config_proto

CPU

GPU 2J 8�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
_output_shapes
: *
T0�

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*(
_output_shapes
:����������*
T0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"B
statefulpartitionedcall_args_2 statefulpartitionedcall_args_2_0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0*?
_input_shapes.
,: : : : :����������: : :::22
StatefulPartitionedCallStatefulPartitionedCall: : : :	 :  : : : : : 
� 
�
B__inference_dense_3_layer_call_and_return_conditional_losses_42457

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:_
Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������k
Tensordot/transpose_1/permConst*
_output_shapes
:*
valueB"       *
dtype0�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	�j
Tensordot/Reshape_1/shapeConst*
valueB"�      *
dtype0*
_output_shapes
:�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
_output_shapes
:	�*
T0�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*'
_output_shapes
:���������*
T0[
Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :�������������������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*4
_output_shapes"
 :������������������*
T0"
identityIdentity:output:0*<
_input_shapes+
):�������������������::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
�!
�
while_body_42929
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*
T0*(
_output_shapes
:�����������
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_1AddV2placeholderadd_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_2AddV2while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes.
,: : : : :����������: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:	 :  : : : : : : : : 
�
�
,__inference_simple_rnn_3_layer_call_fn_43141

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-42403*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42275*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*5
_output_shapes#
!:��������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*5
_output_shapes#
!:�������������������*
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�!
�
while_body_42321
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0%
!biasadd_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*
T0*(
_output_shapes
:�����������
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderTanh:y:0*
element_dtype0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_1AddV2placeholderadd_1/y:output:0*
T0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_2AddV2while_loop_counteradd_2/y:output:0*
T0*
_output_shapes
: �
IdentityIdentity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: �

Identity_4IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes.
,: : : : :����������: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : :	 :  : : : : 
�
�
while_cond_42066
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : ::::  : : : : : : : : :	 
�	
�
,__inference_sequential_3_layer_call_fn_42511
simple_rnn_3_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_3_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-42503*P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_42502*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin

2*4
_output_shapes"
 :�������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::22
StatefulPartitionedCallStatefulPartitionedCall:2 .
,
_user_specified_namesimple_rnn_3_input: : : : : 
�A
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43399
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: N
zeros/mul/yConst*
value
B :�*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :�Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0Q
zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:����������
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0n
TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resourcebiasadd_readvariableop_resource matmul_1_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*9
output_shapes(
&: : : : :����������: : : : : *
T
2
*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_43319*
_num_original_outputs
*
bodyR
while_body_43320*:
_output_shapes(
&: : : : :����������: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: _
while/Identity_4Identitywhile:output:4*(
_output_shapes
:����������*
T0M
while/Identity_5Identitywhile:output:5*
_output_shapes
: *
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*(
_output_shapes
:����������*
T0*
Index0e
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile: :( $
"
_user_specified_name
inputs/0: : 
�
�
,__inference_simple_rnn_3_layer_call_fn_43415
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-42141*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42140*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*5
_output_shapes#
!:��������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
�
�
'__inference_dense_3_layer_call_fn_43456

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*4
_output_shapes"
 :������������������*,
_gradient_op_typePartitionedCall-42463*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42457*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*<
_input_shapes+
):�������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�:
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42016

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: N
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value
B :�_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
value
B :�*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
���������*
dtype0*
_output_shapes
: �
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_mask�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-41680*U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41656*
Tout
2**
config_proto

CPU

GPU 2J 8*<
_output_shapes*
(:����������:����������*
Tin	
2n
TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4^StatefulPartitionedCall*
_num_original_outputs
*
bodyR
while_body_41943*:
_output_shapes(
&: : : : :����������: : : : : *9
output_shapes(
&: : : : :����������: : : : : *
T
2
*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_41942K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0_
while/Identity_4Identitywhile:output:4*(
_output_shapes
:����������*
T0M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
���������*
dtype0a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*(
_output_shapes
:����������e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*5
_output_shapes#
!:�������������������*
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs: : : 
�
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_42475
simple_rnn_3_input/
+simple_rnn_3_statefulpartitionedcall_args_1/
+simple_rnn_3_statefulpartitionedcall_args_2/
+simple_rnn_3_statefulpartitionedcall_args_3*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_3/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_3_input+simple_rnn_3_statefulpartitionedcall_args_1+simple_rnn_3_statefulpartitionedcall_args_2+simple_rnn_3_statefulpartitionedcall_args_3**
config_proto

CPU

GPU 2J 8*
Tin
2*5
_output_shapes#
!:�������������������*,
_gradient_op_typePartitionedCall-42403*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42275*
Tout
2�
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*4
_output_shapes"
 :������������������*,
_gradient_op_typePartitionedCall-42463*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42457*
Tout
2�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:2 .
,
_user_specified_namesimple_rnn_3_input: : : : : 
�
�
,__inference_sequential_3_layer_call_fn_42873

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_42502*
Tout
2**
config_proto

CPU

GPU 2J 8*4
_output_shapes"
 :������������������*
Tin

2*,
_gradient_op_typePartitionedCall-42503�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*4
_output_shapes"
 :������������������*
T0"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : 
�:
�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42140

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0N
zeros/mul/yConst*
_output_shapes
: *
value
B :�*
dtype0_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
dtype0*
_output_shapes
: *
value
B :�s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: �
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0_
strided_slice_2/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:���������*
Index0*
T0�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-41694*U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41675*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*<
_output_shapes*
(:����������:����������n
TensorArrayV2_1/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
���������*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4^StatefulPartitionedCall*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_42066*
_num_original_outputs
*
bodyR
while_body_42067*:
_output_shapes(
&: : : : :����������: : : : : *9
output_shapes(
&: : : : :����������: : : : : *
T
2
K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: _
while/Identity_4Identitywhile:output:4*
T0*(
_output_shapes
:����������M
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*5
_output_shapes#
!:�������������������h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
���������*
dtype0a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*(
_output_shapes
:����������*
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*5
_output_shapes#
!:�������������������*
T0"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2
whilewhile22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
�
�
while_cond_42195
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : :::: : : : : : :	 :  : : 
�
�
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41675

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
��n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:����������H
TanhTanhadd:z:0*(
_output_shapes
:����������*
T0�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:�����������

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:���������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:&"
 
_user_specified_namestates: : : :& "
 
_user_specified_nameinputs
�
�
while_cond_41942
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*?
_input_shapes.
,: : : : :����������: : :::: : : : : : :	 :  : : 
�	
�
1__inference_simple_rnn_cell_3_layer_call_fn_43512

inputs
states_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-41694*U
fPRN
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_41675*
Tout
2**
config_proto

CPU

GPU 2J 8*<
_output_shapes*
(:����������:����������*
Tin	
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:�����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0"
identityIdentity:output:0*F
_input_shapes5
3:���������:����������:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0: : 
� 
�
B__inference_dense_3_layer_call_and_return_conditional_losses_43449

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:_
Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0Y
Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
_output_shapes
: *
T0W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*5
_output_shapes#
!:�������������������*
T0�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������k
Tensordot/transpose_1/permConst*
_output_shapes
:*
valueB"       *
dtype0�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
_output_shapes
:	�*
T0j
Tensordot/Reshape_1/shapeConst*
valueB"�      *
dtype0*
_output_shapes
:�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	��
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :�������������������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*<
_input_shapes+
):�������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_42502

inputs/
+simple_rnn_3_statefulpartitionedcall_args_1/
+simple_rnn_3_statefulpartitionedcall_args_2/
+simple_rnn_3_statefulpartitionedcall_args_3*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_3/StatefulPartitionedCall�$simple_rnn_3/StatefulPartitionedCall�
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCallinputs+simple_rnn_3_statefulpartitionedcall_args_1+simple_rnn_3_statefulpartitionedcall_args_2+simple_rnn_3_statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-42403*P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_42275*
Tout
2**
config_proto

CPU

GPU 2J 8*5
_output_shapes#
!:�������������������*
Tin
2�
dense_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-42463*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42457*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*4
_output_shapes"
 :�������������������
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*4
_output_shapes"
 :������������������*
T0"
identityIdentity:output:0*G
_input_shapes6
4:������������������:::::2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
^
simple_rnn_3_inputH
$serving_default_simple_rnn_3_input:0������������������H
dense_3=
StatefulPartitionedCall:0������������������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
D__call__
*E&call_and_return_all_conditional_losses
F_default_save_signature"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_3", "layers": [{"class_name": "SimpleRNN", "config": {"name": "simple_rnn_3", "trainable": true, "batch_input_shape": [null, null, 2], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 2], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "SimpleRNN", "config": {"name": "simple_rnn_3", "trainable": true, "batch_input_shape": [null, null, 2], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

trainable_variables
	variables
regularization_losses
	keras_api
G__call__
*H&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "simple_rnn_3_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, null, 2], "config": {"batch_input_shape": [null, null, 2], "dtype": "float32", "sparse": false, "name": "simple_rnn_3_input"}}
�	
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
I__call__
*J&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SimpleRNN", "name": "simple_rnn_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, null, 2], "config": {"name": "simple_rnn_3", "trainable": true, "batch_input_shape": [null, null, 2], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 2], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
K__call__
*L&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�
iter

beta_1

beta_2
	decay
learning_ratem:m;m< m=!m>v?v@vA vB!vC"
	optimizer
C
0
 1
!2
3
4"
trackable_list_wrapper
C
0
 1
!2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
"non_trainable_variables

#layers
trainable_variables
	variables
$metrics
regularization_losses
%layer_regularization_losses
D__call__
F_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Mserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
&non_trainable_variables

'layers

trainable_variables
	variables
(metrics
regularization_losses
)layer_regularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�

kernel
 recurrent_kernel
!bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
N__call__
*O&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_rnn_cell_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
5
0
 1
!2"
trackable_list_wrapper
5
0
 1
!2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
.non_trainable_variables

/layers
trainable_variables
	variables
0metrics
regularization_losses
1layer_regularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_3/kernel
:2dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
2non_trainable_variables

3layers
trainable_variables
	variables
4metrics
regularization_losses
5layer_regularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
&:$	�2simple_rnn_3/kernel
1:/
��2simple_rnn_3/recurrent_kernel
 :�2simple_rnn_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
 1
!2"
trackable_list_wrapper
5
0
 1
!2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
6non_trainable_variables

7layers
*trainable_variables
+	variables
8metrics
,regularization_losses
9layer_regularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
&:$	�2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
+:)	�2Adam/simple_rnn_3/kernel/m
6:4
��2$Adam/simple_rnn_3/recurrent_kernel/m
%:#�2Adam/simple_rnn_3/bias/m
&:$	�2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
+:)	�2Adam/simple_rnn_3/kernel/v
6:4
��2$Adam/simple_rnn_3/recurrent_kernel/v
%:#�2Adam/simple_rnn_3/bias/v
�2�
,__inference_sequential_3_layer_call_fn_42511
,__inference_sequential_3_layer_call_fn_42883
,__inference_sequential_3_layer_call_fn_42535
,__inference_sequential_3_layer_call_fn_42873�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_sequential_3_layer_call_and_return_conditional_losses_42863
G__inference_sequential_3_layer_call_and_return_conditional_losses_42475
G__inference_sequential_3_layer_call_and_return_conditional_losses_42708
G__inference_sequential_3_layer_call_and_return_conditional_losses_42488�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_41605�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *>�;
9�6
simple_rnn_3_input������������������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
,__inference_simple_rnn_3_layer_call_fn_43415
,__inference_simple_rnn_3_layer_call_fn_43407
,__inference_simple_rnn_3_layer_call_fn_43149
,__inference_simple_rnn_3_layer_call_fn_43141�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43008
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43399
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43274
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43133�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_dense_3_layer_call_fn_43456�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_3_layer_call_and_return_conditional_losses_43449�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
=B;
#__inference_signature_wrapper_42551simple_rnn_3_input
�2�
1__inference_simple_rnn_cell_3_layer_call_fn_43512
1__inference_simple_rnn_cell_3_layer_call_fn_43501�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_43473
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_43490�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
,__inference_simple_rnn_3_layer_call_fn_43149w! H�E
>�;
-�*
inputs������������������

 
p 

 
� "&�#��������������������
,__inference_sequential_3_layer_call_fn_42883t! D�A
:�7
-�*
inputs������������������
p 

 
� "%�"�������������������
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43133�! H�E
>�;
-�*
inputs������������������

 
p 

 
� "3�0
)�&
0�������������������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_42863�! D�A
:�7
-�*
inputs������������������
p 

 
� "2�/
(�%
0������������������
� �
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_43473�! ]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p
� "T�Q
J�G
�
0/0����������
%�"
 �
0/1/0����������
� �
1__inference_simple_rnn_cell_3_layer_call_fn_43501�! ]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p
� "F�C
�
0����������
#� 
�
1/0�����������
 __inference__wrapped_model_41605�! H�E
>�;
9�6
simple_rnn_3_input������������������
� ">�;
9
dense_3.�+
dense_3�������������������
L__inference_simple_rnn_cell_3_layer_call_and_return_conditional_losses_43490�! ]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p 
� "T�Q
J�G
�
0/0����������
%�"
 �
0/1/0����������
� �
1__inference_simple_rnn_cell_3_layer_call_fn_43512�! ]�Z
S�P
 �
inputs���������
(�%
#� 
states/0����������
p 
� "F�C
�
0����������
#� 
�
1/0�����������
,__inference_sequential_3_layer_call_fn_42511�! P�M
F�C
9�6
simple_rnn_3_input������������������
p

 
� "%�"�������������������
G__inference_sequential_3_layer_call_and_return_conditional_losses_42708�! D�A
:�7
-�*
inputs������������������
p

 
� "2�/
(�%
0������������������
� �
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43399�! O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "3�0
)�&
0�������������������
� �
'__inference_dense_3_layer_call_fn_43456j=�:
3�0
.�+
inputs�������������������
� "%�"�������������������
,__inference_sequential_3_layer_call_fn_42535�! P�M
F�C
9�6
simple_rnn_3_input������������������
p 

 
� "%�"�������������������
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43008�! H�E
>�;
-�*
inputs������������������

 
p

 
� "3�0
)�&
0�������������������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_42475�! P�M
F�C
9�6
simple_rnn_3_input������������������
p

 
� "2�/
(�%
0������������������
� �
#__inference_signature_wrapper_42551�! ^�[
� 
T�Q
O
simple_rnn_3_input9�6
simple_rnn_3_input������������������">�;
9
dense_3.�+
dense_3�������������������
G__inference_sequential_3_layer_call_and_return_conditional_losses_42488�! P�M
F�C
9�6
simple_rnn_3_input������������������
p 

 
� "2�/
(�%
0������������������
� �
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_43274�! O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "3�0
)�&
0�������������������
� �
B__inference_dense_3_layer_call_and_return_conditional_losses_43449w=�:
3�0
.�+
inputs�������������������
� "2�/
(�%
0������������������
� �
,__inference_simple_rnn_3_layer_call_fn_43407~! O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "&�#��������������������
,__inference_simple_rnn_3_layer_call_fn_43141w! H�E
>�;
-�*
inputs������������������

 
p

 
� "&�#��������������������
,__inference_simple_rnn_3_layer_call_fn_43415~! O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "&�#��������������������
,__inference_sequential_3_layer_call_fn_42873t! D�A
:�7
-�*
inputs������������������
p

 
� "%�"������������������