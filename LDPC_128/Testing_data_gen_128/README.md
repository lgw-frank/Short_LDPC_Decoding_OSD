Claim:The credits of some code snippets are contributed to their original owners definitely, we promise to  reuse them for academic purpose only. 
       Surely we are not intended to infringe any intellectual property right, please contact us for immediate corrections if any. 
       
The entry function is defined in Main_test.py file. To customize it to your scenario, just follow the listed steps 
(Assume Spyder IDE run in Anaconda fo Win10):
1) Check and make sure all Python packages in each file of subdirectory are installed; 
2) In line 23 of Main_test.py, modify the arguments in line 23. For code snippet like this:
   
  sys.argv = "python 2.0 3.0 6 100 1000 CCSDS_ldpc_n128_k64.alist".split()
   
substitue your appropriate evaluations for those arguments of line 23, here SNR=2.0dB-3.0dB with increment 0.2dB since the total
number of SNR points are 6. Again, the largest number of batches each of which is of size 100 is 1000, the last argument is 
the same parity check matrix file with the other modules. In line 51 like this:
GL.set_map('portion_dis', '0.05 0.075 0.2 0.5 0.75 1.0')
we allocate different proportions for the testing data samples for various SNR points to save disk space.

3) In line 49, decide whether to use all-zero codewords or not depending on your taste (Here 'False'),
   GL.set_map('ALL_ZEROS_CODEWORD_TESTING', False)
   
4) Click the 'Run file' button in the menu of Spyder, it is expected a ist of testing samples for the specified SNRs within a few minutes for the
    examplary arguments after the reminding output of the last line of Main_train.py:
            print("Data for testing generated successfully!")
   
Notice: According to the for loop since line 93, we can check if the data files, or the ultimate outputs of this module,
with designated names are successfully generated in the designated directory. The generated data files will be the input data 
to the next module in the directory 'Ldpc_128_testing'. 
