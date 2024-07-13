Claim:The credits of some code snippets are contributed to their original owners definitely, we promise to  reuse them for academic purpose only. 
       Surely we are not intended to infringe any intellectual copyright, please contact us for immediate corrections if any. 
       
The entry function is defined in Main_train.py file. To customize it to your scenario, just follow the listed steps 
(Assume Spyder IDE run in Anaconda fo Win10):
1) Check and make sure all Python packages in each file of subdirectory are installed; 
2) In line 17-22 of Main_train.py, modify the arguments in line 17. For code snippet like this:
   
    sys.argv = "python 2.7 2.7 100 1000 CCSDS_ldpc_n128_k64.alist".split()
   
    GL.set_map('snr_lo', float(sys.argv[1]))
   
    GL.set_map('snr_hi', float(sys.argv[2]))
   
    GL.set_map('batch_size', int(sys.argv[3]))
   
    GL.set_map('training_batch_number', int(sys.argv[4]))
   
    GL.set_map('H_filename', sys.argv[5])
   
substitue your appropriate evaluations for those arguments of line 17, each of which generates a map in line 18-22 separately
for later populating later variables.
In our scenario, the lower limit and upper limit of SNRs are the same 2.7dB for the AWGN channel, denoting sampling from one
SNR point at 2.7dB, otherwise an even blended samples from SNRs of lower limit to upper limit are generated with the increment of
0.1dB. Then there're total 1000 training batches  each of which is size of 100 to be generated. The called parityc
check matrix of the code is named *.alist (here CCSDS_ldpc_n128_k64.alist for LDPC (128,64) code). Some othe codes are included,
for more codes: see https://rptu.de/channel-codes/ml-simulation-results

3) In line 30, decide whether to use all-zero codewords or not depending on your taste (Here 'False'),
   GL.set_map('ALL_ZEROS_CODEWORD_TRAINING', False)
   
4) Click the 'Run file' button in the menu of Spyder, it is expected a training file is generated  within 5 minutes for the
    examplary arguments after the reminding output of the last line of Main_train.py:
            print("Data for training generated successfully!")
   
Notice: According to line 55, we can check if the data file, or the ultimate output of this module,
with designated name is successfully generated in the designated directory.
