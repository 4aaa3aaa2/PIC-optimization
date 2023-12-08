import  numpy as np
import  os 
import time
import random 
import copy
import matplotlib.pyplot as plt 


global N
N=4

shifter_num=int((N*(N-1)/2))+N
N_try=10*shifter_num

Umatrix_file=f'TargetMatrices_N={N}.npy'
targetU_list=np.load(Umatrix_file)

#
# TODO the part of util functions
#
def list_multiply(list_imput):
    #TODO return the product of all elements in the given list
    U=np.eye(N)
    for i in list_imput:
        U=np.matmul(i,U)
    return U

def pnp_multiply(list_input):
    U = np.eye(N)
    for i in range(len(list_input)):
        U = np.matmul(list_input[i].matrix,U)
    return U

def fidelity(target_matrix, exp_matrix):
    #TODO calculate the fidelity of the experimental matrix and tazrget matrix
    # target: target matrix; exp: experimental matrix
    a = np.matrix.trace(np.transpose(np.conj(target_matrix)) @ exp_matrix)
    b = np.matrix.trace(np.transpose(np.conj(exp_matrix)) @ exp_matrix)
    return np.square(abs(a/np.sqrt(N*b)))

def angle_shifter(angle):
    #TODO for a given phase, change it into the range [0,2pi)
    while angle < 0:
        angle += 2.0*np.pi
    while angle > 2*np.pi:
        angle -= 2.0*np.pi
    return angle

#
# TODO define the class of MZI tranbsfer matrix and inverse matrix
#

class T_MZI:
    # this class defines the transfer matrix of an mzi phase shifter
    def __init__(self,phase_int=0, phase_ext=0, i=0):
        self.zeta=phase_int
        self.fai=phase_ext
        self.i=i
       
        self.matrix=np.eye(N,dtype=np.complex128)
        a=np.exp(1.0j*(self.zeta+np.pi)*0.5)
        self.matrix[i,i]=a*np.exp(1.0j*self.fai)*np.sin(self.zeta/2)
        self.matrix[i,i+1] = a*np.cos(self.zeta/2)
        self.matrix[i+1,i] = a*np.exp(1.0j*self.fai)*np.cos(self.zeta/2)
        self.matrix[i+1,i+1] = a*-np.sin(self.zeta/2)
    
    def MZI_get__angle(self, matrix_input,i):#TODO: can this really work?
        self.matrix=matrix_input
        self.i=i
        self.zeta=angle_shifter(np.angle(matrix_input[i, i + 1]) * 2.0 - np.pi)
        self.fai=angle_shifter(np.angle(matrix_input[i, i]) - np.angle(matrix_input[i, i + 1]))


class T_inverse_MZI:
    # this class defines the inverse transfer matrix of an mzi phase shifter
    def __init__(self, phase_int=0, phase_ext=0, i=0):
        self.zeta = phase_int
        self.fai = phase_ext
        self.i = i
        # transfer matrix
        self.matrix = np.eye(N,dtype=np.complex128)
        a = np.exp(-1.0j*(self.zeta+np.pi)/2)
        self.matrix[i,i] = a*np.exp(-1.0j*self.fai)*np.sin(self.zeta/2)
        self.matrix[i,i+1] = a*np.exp(-1.0j * self.fai) * np.cos(self.zeta / 2)
        self.matrix[i+1,i] = a*np.cos(self.zeta / 2)
        self.matrix[i+1,i+1] = a*-np.sin(self.zeta/ 2)
    
    def MZI_get_angle(self,matrix_input,i):
        self.matrix = matrix_input
        self.i = i
        self.zeta = angle_shifter(np.angle(matrix_input[i + 1, i]) * -2.0 - np.pi)
        self.fai  = angle_shifter(-(np.angle(matrix_input[i, i]) - np.angle(matrix_input[i + 1, i])))

class lossy_T_MZI:
    # this class defines the lossy transfer T of each MZI
    def __init__(self, phase_int=0, phase_ext=0, i=0, modulation_loss=0, passive_loss=0):
        self.zeta=phase_int
        self.fai=phase_ext
        self.i=i
        self.mod_loss=-modulation_loss
        self.pas_loss=-passive_loss

        self.matrix=np.eye(N,dtype=np.complex128)

        A=10**((self.mod_loss*self.zeta)/(20*np.pi))*np.exp(1j*self.zeta)
        B=10**((self.mod_loss*self.fai)/(20*np.pi))*np.exp(1j*self.fai)
        #A=10**((self.mod_loss*self.zeta)/(20*np.pi))*np.exp(1j*self.zeta)
        #B=10**((self.mod_loss*self.fai)/(20*np.pi))*np.exp(1j*self.fai)
        coeff=0.5*10**(self.pas_loss*1/10)

        self.matrix[i,i]=coeff*(A*B-B)
        self.matrix[i,i+1]=coeff*(1j*A+1j)
        self.matrix[i+1,i]=coeff*(1j*A*B+1j*B)
        self.matrix[i+1,i+1]=coeff*(1-A)
#
# TODO define the process of unitary matrix decomposition
#

class PNP:
    def __init__(self):

        self.T_list = []
        self.left_T_list1 = []
        self.right_T_list1 = []
        self.left_T_list2 = []
        self.right_T_list2 = []
        self.left_T_list3 = []

        self.D_ph=np.zeros(N)
        self.D = np.eye(N)
        self.StageMatrix_list = []
        self.lossy_T_list=[]

    def null_by_col(self,U,i,j):
        fai = angle_shifter(np.angle(U[i,j])-np.angle(U[i,j+1])-np.pi)
        zeta = np.arctan(abs(U[i,j+1])/abs(U[i,j]))*2
        T_mzi = T_inverse_MZI(zeta,fai,j)
        self.right_T_list1.append(T_mzi)
        return np.matmul(U,T_mzi.matrix)
    
    def null_by_row(self,U,i,j):
        fai = angle_shifter(np.angle(U[i,j])-np.angle(U[i-1,j]))
        zeta = np.arctan(abs(U[i-1,j])/abs(U[i,j]))*2
        T_mzi = T_MZI(zeta,fai,i-1)
        self.left_T_list1.append(T_mzi)
        return np.matmul(T_mzi.matrix,U)
    
    def D_swap(self,T_inv, D):
        zeta = T_inv.zeta
        fai = T_inv.fai
        i = T_inv.i
        a = np.angle(D[i+1,i+1])
        b = angle_shifter(np.angle(D[i,i])-a)
        T = T_MZI (zeta,b,i)
        D_new = D
        D_new[i,i] = np.exp(1j*(a-zeta-np.pi-fai))
        D_new[i+1,i+1] = np.exp(1j*(a-zeta-np.pi))
        return T, D_new

    def decompose(self,U):
        self.T_list.clear()
        self.left_T_list1.clear()
        self.left_T_list2.clear()
        self.left_T_list3.clear()
        
        self.right_T_list1.clear()
        self.right_T_list2.clear()

        # null the elements in U matrix, 
        # and record the current mzi T matrix into <<left T list 1>> and <<right T list 1>>
        for i in range(N-1):
            if i%2 == 0:
                for j in range(i+1):
                    U=self.null_by_col(U,N-1-j,i-j)
            else:
                for j in range(i+1):
                    U=self.null_by_row(U,N+j-i-1,j)
        self.D=U

        # change the order of mzi T matrices in list1 
        # and switch those matrices from T^(-1) to T, or from T to T^(-1) 
        # << right T list 1>> --> <<right T list 2>>
        for i in range(len(self.right_T_list1))[::-1]: 
            zeta_temp = self.right_T_list1[i].zeta
            fai_temp = self.right_T_list1[i].fai
            i_temp = self.right_T_list1[i].i
            self.right_T_list2.append(T_MZI(zeta_temp,fai_temp,i_temp))
        
        # <<left T list 1>> --> <<left T list 2>>
        for i in range(len(self.left_T_list1)): 
            zeta_temp = self.left_T_list1[i].zeta
            fai_temp = self.left_T_list1[i].fai
            i_temp = self.left_T_list1[i].i
            self.left_T_list2.append(T_inverse_MZI(zeta_temp,fai_temp,i_temp))
        
        # <<left T list 2>> --> <<left T list 3>>
        # also get the final D matrix
        for i in range(len(self.left_T_list2))[::-1]:
            T_temp,self.D = self.D_swap(self.left_T_list2[i],self.D)
            self.left_T_list3.append(T_temp)
        
        self.left_T_list3.reverse()
        for i in range(len(self.right_T_list2))[::-1]:
            self.T_list.append(self.right_T_list2[i])
        for i in range(len(self.left_T_list3))[::-1]:
            self.T_list.append(self.left_T_list3[i])

        for i in range(N):
            self.D_ph[i]=np.angle(self.D[i,i])
    
    def Transfer_matrix(self):
        return np.matmul(self.D,pnp_multiply(self.T_list))
    
    def phase_export(self, phase_array, d_phase_array):
        phase_array=np.zeros((int(N/2),N,2))
        d_phase_array=np.zeros(N)
        if N % 2 == 0:
            num = 0
            for i in range(int(N/2)):
                for j in range(2*i+1):
                    phase_array[i-int((j+1)/2), j, 0] = self.T_list[num].zeta
                    phase_array[i-int((j+1)/2), j, 1] = self.T_list[num].fai
                    num += 1
            for i in range(int(N/2)-1):
                for j in range(N-2-2*i):
                    phase_array[int(N/2)-1-int((j+1)/2), j+2+2*i, 0] = self.T_list[num].zeta
                    phase_array[int(N/2)-1-int((j+1)/2), j+2+2*i, 1] = self.T_list[num].fai
                    num += 1
        else:
            num = 0
            for i in range(int(N/2)):
                for j in range(2*i+1):
                    phase_array[i-int((j + 1) / 2), j, 0] = self.T_list[num].zeta
                    phase_array[i-int((j + 1) / 2), j, 1] = self.T_list[num].fai
                    num += 1
            for i in range(int(N/2)):
                for j in range(N-1-2*i):
                    phase_array[int(N/2)-1-int(j/2), 2*i+1+j, 0] = self.T_list[num].zeta
                    phase_array[int(N/2)-1-int(j/2), 2*i+1+j, 1] = self.T_list[num].fai
                    num += 1
        for i in range(N):
            d_phase_array[i] = self.D_ph[i]
        return phase_array, d_phase_array

    # given the list of original transfer matrices of MZIs, add modulation loss and passive loss to 
    # all these T matrices.
    def add_loss_T_list(self, modulation_loss, passive_loss):
        self.lossy_T_list.clear()
        for i in range(len(self.T_list)):
            Tmzi= self.T_list[i]
            zeta_temp=Tmzi.zeta
            fai_temp=Tmzi.fai
            location_i=Tmzi.i

            lossy_Tmzi=lossy_T_MZI(zeta_temp,fai_temp,location_i,modulation_loss,passive_loss)
            self.lossy_T_list.append(lossy_Tmzi)
        
    def lossy_Transfer_matrix(self):
        return np.matmul(self.D,pnp_multiply(self.lossy_T_list))


#
# TODO the SA process
# 
def b_update(b):
    if b>5000: 
        b-=3
    elif b>2000:
        b-=2
    else:
        b-=1
    return b 
def T_update(T,T_start):
    if T>0.5*T_start:
        T/=1.6
    elif T<0.5*T_start and T>0.2*T_start:
        T/=1.3
    elif T<0.2*T_start and T>0.1*T_start:
        T/=1.2
    else:
        T/=1.05
    return T
    
def f_accept_condttion(f_new, f, T):
    #if  np.exp((f_new-f)/T)> max(random.uniform(0.9*np.exp((f_new-f)/T) ,2), random.uniform(0.9999,1) , 0.99999):
    #if np.exp((f-f_new)/T)-1> random.uniform(0,0.1):
    
    if np.exp((f-f_new)/T) > random.uniform(0.9*np.exp((f-f_new)/T),min(2*np.exp((f-f_new)/T),1)):
        return True
    else:
        return False
    
def simulated_annealing(accuracy,b_start,T_start,T_end,lossy_T_list_input,D_input, U_target,modulation_loss, passive_loss, f):
    a=0
    #better_f=0
    better_f = f
    #best_f=0
    best_f=f
    count=0
    f_rec=[]
    best_f_rec=[]
    b= int(b_start/accuracy)
    T=T_start
    while b > 0 and T > T_end:
        N_jump=0
        for i in range(N_try):
            random_num=random.randrange(shifter_num)
            if random_num < shifter_num-N:
            # if an MZI fs is chosen
                n = random.randrange(N)
                int_or_ext = random.randrange(2)

                lossy_T_temp=copy.deepcopy(lossy_T_list_input[n]) # reserve the current value first
                # calculate the new value
                if int_or_ext == 0: 
                    this_phase = lossy_T_list_input[n].zeta
                    this_phase += random.randint(-b,b)*accuracy
                    this_phase = angle_shifter(this_phase)

                    this_fai=lossy_T_list_input[n].fai
                    this_i=lossy_T_list_input[n].i 
                    lossy_T_list_input[n]=lossy_T_MZI(this_phase, this_fai, this_i, modulation_loss, passive_loss)
                else:
                    this_phase=lossy_T_list_input[n].fai
                    this_phase += random.randint(-b,b)*accuracy
                    this_phase = angle_shifter(this_phase)

                    this_zeta=lossy_T_list_input[n].zeta
                    this_i=lossy_T_list_input[n].i
                    lossy_T_list_input[n]=lossy_T_MZI(this_zeta, this_phase, this_i, modulation_loss, passive_loss)

            else:
            # if one shifter in the last shifter array is chosen
                s=random_num-(shifter_num-N)
                ph_temp=copy.deepcopy(D_input[s,s])
                this_phase = np.angle(D_input[s,s])
                this_phase += random.randint(-b,b)*accuracy
                this_phase = angle_shifter(this_phase)
                D_input[s,s] = np.exp(1j*this_phase) 
            
            lossy_Uexp_temp = np.matmul(D_input, pnp_multiply(lossy_T_list_input))
            f_temp = fidelity(U_target, lossy_Uexp_temp)
            
            if f_temp>better_f:
                better_f = f_temp
                if better_f > best_f:
                    best_f=better_f
                    best_T_List=copy.deepcopy(lossy_T_list_input)
                    best_D=copy.deepcopy(D_input)
                if N_jump > 0:
                    N_jump -= 1
                f_rec.append(better_f)
                best_f_rec.append(best_f)
                a += 1
            else:
                #if f_accept_condttion(f_temp, f, T):
                if np.exp((f_temp - f) / T) > random.random():
                #if np.exp((f_temp - f) / T) > random.uniform(0.9*np.exp((f_temp - f) / T),1) and N_jump<5:
                    better_f=f_temp
                    N_jump+=1
                    f_rec.append(better_f)
                    best_f_rec.append(best_f)
                    a+=1
                else:
                    if random_num < shifter_num - N:
                        lossy_T_list_input[n] = lossy_T_temp
                    else: 
                        D_input[s,s]= ph_temp      
            
            count+=1
  
        if a>=N_try*0.5 :
            T =  T_update(T,T_start)
        else:
            b=b_update(b)
            #T/=1.0005
    #print('over')
    #return better_f,best_f,best_T_List,best_D,f_rec, best_f_rec, lossy_T_list_input, D_input,count  #TODO this is used for main1()
    #return best_f #TODO this is used for main2()
    return best_f, f_rec, best_f_rec

#
# TODO the main flow of calculation 
#

passive_loss=0
modulation_loss_values_list=[0,0.3,0.6,0.9,1.2, 1.5]


def T_list_phase(T_list):
    zeta_list=[]
    fai_list=[]
    for i in T_list:
        zeta_list.append(i.zeta)
        fai_list.append(i.fai)
    return zeta_list, fai_list



def main1():
    # see the detailed optimized result and process of one matrix
    modulation_loss=1.5
    org_f_list=[]
    sa_best_f_list=[]
    U_matrix_target=targetU_list[4]
    pnp=PNP()
    pnp.decompose(U_matrix_target)
    
    pnp.add_loss_T_list(modulation_loss,0)
    Uexp_without_SA=pnp.lossy_Transfer_matrix()
    F_without_SA=fidelity(U_matrix_target,Uexp_without_SA)
    zlist,flist=T_list_phase(pnp.lossy_T_list)
    print('lossy f',F_without_SA)
    #print('see T  zeta series', zlist)
    #print('see T fai series', flist)
    lossy_T_list_SA = copy.deepcopy(pnp.lossy_T_list)
    D_SA = copy.deepcopy(pnp.D)
    org_f_list.append(F_without_SA)
    SA_best_f , SA_f_rec, best_f_rec= simulated_annealing(0.000001, 4, 2000, 0.0001, lossy_T_list_SA, D_SA, U_matrix_target, modulation_loss, 0, F_without_SA)
    #org_f_list.append(F_without_SA)
    #print('f after sa', SA_f)
    print('best f',SA_best_f)
    #sazlist, saflist=T_list_phase(SA_T_list)
    #sa_best_f_list.append(SA_best_f)
    #print('see SA T zeta series', sazlist)
    #print('see SA T fai series', saflist)

    #best_zeta,best_fai=T_list_phase(best_T)
    #print('best zeta',best_zeta)
    #print('best fai',best_fai)
    #print('f rec', SA_f_rec)
    #print(len(SA_f_rec))
    plt.plot(SA_f_rec)
    plt.plot(best_f_rec)
    
    plt.show()
    #print('count',count)
    return org_f_list,sa_best_f_list

#start_time=time.time()
main1()
#print('time cost:',time.time()-start_time)


import pandas as pd


def main2():
    # get the result of all matrices
    df=pd.DataFrame(modulation_loss_values_list,columns=['mod loss'])
    df1=copy.deepcopy(df)
    df2=copy.deepcopy(df)
    for i in range(100):  
        print('curr i', i) 
        org_f_list=[]
        better_res_list=[]

        U_matrix_target=targetU_list[i]
        pnp=PNP()
        pnp.decompose(U_matrix_target)
        for loss in modulation_loss_values_list:
            pnp.add_loss_T_list(loss,0)
            Uexp_without_SA=pnp.lossy_Transfer_matrix()
            F_without_SA=fidelity(U_matrix_target,Uexp_without_SA)
            lossy_T_list_SA = copy.deepcopy(pnp.lossy_T_list)
            D_SA = copy.deepcopy(pnp.D) 
            SA_best_f = simulated_annealing(0.000001, 4, 2000, 0.0001, lossy_T_list_SA, D_SA, U_matrix_target, loss, 0, F_without_SA)
            org_f_list.append(F_without_SA)
            better_res = max(SA_best_f, F_without_SA)
            better_res_list.append(better_res)
        df1 = pd.concat([df1, pd.DataFrame(org_f_list,columns=[f'{i}'])], axis=1)
        df2 = pd.concat([df2, pd.DataFrame(better_res_list,columns=[f'{i}'])], axis=1)  
           
            
    return df1,df2

#df1, df2=main2()
#df1.to_csv(f'original f N={N}.csv', index=False)
#df2.to_csv(f'improved f N={N}.csv', index=False)
