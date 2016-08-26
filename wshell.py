# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:45:19 2016

@author: Rachael Mansbach
"""
import mdtraj as md,numpy as np, networkx as nx, imp, re
mk = imp.load_source('mk','/home/rachael/cluster/home/mansbac2/coarsegraining/code/markov/markov.py')

def groupFromNdx(ndxfile,gname):
    #find the indices of a group with a particular name from a gromacs .ndx file
    #gname should be something like [ Protein ]    
    nfile = open(ndxfile,'r')
    group = []
    flag = ''
    while flag!=gname:
        flag = nfile.readline()
        flag = flag.rstrip()
    while True:
        nums = nfile.readline()
        if nums[0] == '[':
            return group
        nums = nums.split()
        for n in nums:
            ind = int(n) - 1 #gromacs is 1-indexed
            group.append(ind)
    nfile.close()
    return group

def groTop(fname,nlines):
    #read in the topology from a gro file for ease of formatting output
    #generally use this with a protein-only file
    f = open(fname,'r')
    top = []
    for i in range(nlines-1):
        line = f.readline()
        if (i > 1):
            spline = line.split()
            top.append([spline[0],spline[1]])
    f.close()
    return top
    
def pairsToCheck(v1,v2):
    #given two numpy arrays, produces another numpy array consisting of all combinations
    # of the two provided
    v = np.zeros([len(v1)*len(v2),2])
    for i in range(len(v1)):
        for j in range(len(v2)):
            v[len(v2)*i+j,0] = v1[i]
            v[len(v2)*i+j,1] = v2[j]
    return v
    
def writeProtAndWat(ptop,pinds,pxyz,top,wxyz,winds,box,fname,fstyle):
    f = open(fname,fstyle)
    f.write('Protein in water\n')
    #f.write('{0}\n'.format(len(pxyz)/3+len(wxyz)/3))
    f.write('{0}\n'.format(top.n_atoms))    
    ri = -1
    pind = 0
    allatoms = range(top.n_atoms)
    
    for pi in range(0,len(pxyz)/3):
        
        allatoms.remove(pi)
        rname = ptop[pi][0]
        aname = ptop[pi][1]
        ri = re.search('\d+',rname)
        rname = rname[ri.end(0):len(rname)] #split apart residue id and residue name for correct output
        
        ri = ri.group(0)
        pind = pinds.index(pi)
        pxyza = pxyz[3*pind:3*pind+3]        
        f.write('%5d%-5s%5s%5d% 8.3f% 8.3f %8.3f\n' % (int(ri),rname,aname,pi+1,pxyza[0],pxyza[1],pxyza[2]))
    rind = int(ri)+1#track residue number
    aind = pi+1 #track atom number
    for wind in range(0,len(wxyz)/3):
        rname = 'SOL'
        allatoms.remove(winds[wind])
        at = top.atom(winds[wind])
        at = str(at).split('-')[1]
        if at == 'O':
            aname = 'OW'
        elif at == 'H1':
            aname = 'HW1'
        elif at == 'H2':
            aname = 'HW2'
        else:
            break #something went wrong
        wxyza = wxyz[3*wind:3*wind+3]
        f.write('%5d%-5s%5s%5d% 8.3f% 8.3f %8.3f\n' % (rind,rname,aname,aind+1,wxyza[0],wxyza[1],wxyza[2]))
        rind+=1
        aind+=1
    
    #write dummy atoms so each frame has the same number of atoms in it
    af = 0
    for dind in allatoms:
        rname = 'SOL'
        if af == 0:
            aname = 'OW'
            af+=1
        elif af == 1:
            aname = 'HW1'
            af+=1
        else:
            aname = 'HW2'
            af = 0
        f.write('%5d%-5s%5s%5d% 8.3f% 8.3f %8.3f\n' % (rind,rname,aname,aind+1,2*box[0],0,0))
        rind+=1
        aind+=1
        
    f.write(' {0} {1} {2}\n'.format(box[0],box[1],box[2]))
    f.close()

def addBonds(topfile,top):
    #since mdtraj can't read .top files, this function will add bonds to the topology for every bond in an itp file
    tf = open(topfile,'r')
    flag = ''
    
    while flag!= '[ bonds ]':
        flag = tf.readline()
        flag = flag.rstrip()
    while True:
        line = tf.readline()
        if len(line) > 1:
            if line[0] == '[':
                return
            if line[0] != ';':
                spline = line.split()
                bind1 = int(spline[0])-1 #correct for gromacs 1-indexing
                bind2 = int(spline[1])-1
                at1 = top.atom(bind1)
                at2 = top.atom(bind2)
                top.add_bond(at1,at2)

    tf.close()

def moveShell(frame,top,ptop,cutoff,ndxfile,outgro,wstyle):
    #function that makes protein whole and moves neighboring water atoms with it
    #basically, takes an atom, its bonded neighbors & its neighboring water atoms
    #and then fixes their coordinates over PBCs
    prot = np.array(groupFromNdx(ndxfile,'[ Protein ]'))
    pots = [prot[0]]
    winds = []
    wxyzs = []
    pxyzs = []
    pinds = [prot[0]]
    box = frame.unitcell_lengths[0]
    wsel = top.select("water and name O")
    bgraph = top.to_bondgraph()
    pxyzs0 = frame.xyz[0,prot]
    pxyzs0 = pxyzs0.reshape(1,np.shape(pxyzs0)[0]*np.shape(pxyzs0)[1])[0]
    #writeProtAndWat(ptop,range(len(pxyzs0)),pxyzs0,top,[],[],box,'test0.gro','w')
    #fix coords of first atom, put them in pxyzs
    pxyz = frame.xyz[0,prot[0]]
    pxyz = mk.fixCoords(pxyz.copy(),pxyz.copy(),box)
    fixed = [prot[0]]
    pxyzs = np.hstack([pxyzs,pxyz])  
    #also maybe you should update the reference locations...you know, just MAYBE
    frame.xyz[0,prot[0]] = pxyz
    while len(pots) > 0:
        
        
        curr = pots.pop()
        cxyz = frame.xyz[0,curr]
        wind = md.compute_neighbors(frame,cutoff,np.array([curr]),wsel,True)[0].tolist() #get neighboring water atoms to current atom
        #add all bonded hydrogens, which just means add w+1,w+2 for all w in wind
        #but they also need to be IN ORDER, so they need to be right after w
                
        windcopy = list(wind)
        for w in windcopy:
            wind.insert(wind.index(w)+1,w+1)
            wind.insert(wind.index(w+1)+1,w+2)

        windcopy = list(wind)            
        for w in windcopy:
            if w in fixed:
                wind.remove(w)
            else:
                fixed.append(w)
        wxyz = frame.xyz[0,wind]
        
        winds = np.hstack([winds,wind])
        
        neighs = bgraph.neighbors(top.atom(curr))
        xyzs = []
        n = 0
        ninds = []
        for neigh in neighs: #collect all neighbors and their locations
            if not(neigh.index in fixed):
                pots.append(neigh.index)
                fixed.append(neigh.index)
                pinds.append(neigh.index)
                xyzs = np.hstack([xyzs,frame.xyz[0,neigh.index]])
                n+=1
                ninds.append(neigh.index)
        wxyzr = wxyz.reshape([1,np.shape(wxyz)[0]*np.shape(wxyz)[1]])
        xyzs = np.hstack([xyzs,wxyzr[0]])
        fixxyzs = mk.fixCoords(xyzs.copy(),cxyz.copy(),box)
        
        if n > 0:
            #we have neighbors to be added to pxyzs
            pxyzs = np.hstack([pxyzs,fixxyzs[0:3*(n)]])
            #ALSO UPDATE THE FUCKING REFERENCE LOCATIONS YOU TWAT
            for ni in range(n):
                nxyz = fixxyzs[3*ni:3*ni+3]
                frame.xyz[0,ninds[ni]] = nxyz;
            wxyzs = np.hstack([wxyzs,fixxyzs[3*(n):len(fixxyzs)]])
        else:
            #no neighbors, just water
            wxyzs = np.hstack([wxyzs,fixxyzs])
        windslist = []
        for w in winds:
            windslist.append(int(w))
    writeProtAndWat(ptop,pinds,pxyzs,top,wxyzs,windslist,box,outgro,wstyle)

def minDistsT(traj,ndxfile,outfname):
    #find the minimum distance of any water to the backbone at each time step and write out to file
    
    o = open(outfname,'w')
    bb = groupFromNdx(ndxfile,'[ Backbone ]')
    woxsel = traj.topology.select("water and name O")
    apairs = pairsToCheck(np.array(bb),woxsel)
    minds = np.zeros(traj.n_frames)
    for t in range(traj.n_frames):
        print t
        ds = md.compute_distances(traj[t],apairs)
        mds = min(ds[0])
        o.write('{0}\t{1}\n'.format(t,mds))
        minds[t] = mds
    o.close()
    return minds
    
def hbond(traj,donor,acceptor,hydrogen,angle):
    #function that returns whether a hydrogen bond exists between a given set of 
    #candidates for the donor, the acceptor, and the hydrogen
    #assumes that the donor-acceptor distance is already within the correct limit
    #so simply checks the angle
    #angle should be in radians
    dhang = md.compute_angles(traj,np.array([[donor,hydrogen,acceptor]]))
    
    return (dhang[0,0] > angle)
    
class HBNode:
    #a node for an hbgraph
    #it carries a type and an index
    #available types are B for backbone (the central node), C for carbon, N for nitrogen, and W for water
    #the index for B is a dummy -1, for all the others should be the index of their atom in the main trajectory
    #water uses the index of its Oxygen atom
    def __init__(self,Type,index):
        self.type = Type
        self.index = index
    
    def __str__(self):
        return '({0},{1})'.format(self.type,self.index)
        
    def __repr__(self):
        return '({0},{1})'.format(self.type,self.index)

def bbnparse(frame,hbg,bbns,naccbool,woxsel,hbcut,Ninds,wingraph,ningraph,hbang,centNode):
    #make this its own function so we can look at this whole thing more easily
    #this is the part that does the backbone nitrogens & their possible h bonds
    for bbn in bbns:
        #(*)(i) water - if so, add water to graph
        wns = md.compute_neighbors(frame,hbcut,np.array([bbn]),woxsel)[0] #candidate oxygen acceptors
        #I ASSUME AT THIS POINT THAT THE NEXT ATOM IN THE TOPOLOGY FROM bbn IS THE BONDED HYDROGEN (OR THE NEXT THREE FOR THE END RESIDUES)
        #SHOULD RETURN ERROR IF IT'S NOT A HYDROGEN
        nns = md.compute_neighbors(frame,hbcut,np.array([bbn]),np.array(Ninds))[0]
        if (bbn == min(bbns)): 
            #then we are an end residue with a nitrogen + 3 Hs
            for j in range(3):
                hi = bbn+j+1
                h = top.atom(hi)
                if h.name!='H{0}'.format(j+1):
                    #sanity check for hydrogens we're going to use in bonds
                    print "At {0} (end res), did not find a bonded hydrogen.\n".format(bbn)
                    return -1
                else:
                    for wn in wns:
                        hbool = hbond(frame,bbn,wn,hi,hbang)
                        if hbool:
                            wnode = HBNode('W',wn)
                            hbg.add_node(wnode)
                            wingraph.append(wn)
                            hbg.add_edge(centNode,wnode)
                            
                            #^^THIS PART IS SANITY-CHECK APPROVED (picked up all and no more of the Hbonds that VMD did in one particular frame--should keep checking as I go on with different frames)
                    for N in nns:
                        #(ii) if SC ns are an acceptor, the SC nitrogens - either way add SC nitrogens to graph
                        #the SC N indices are in Ninds
                        nnode = HBNode('SCN',N)
                        hbg.add_node(nnode)                        
                        if naccbool:
                            hbool = hbond(frame,bbn,N,hi,hbang)
                            if hbool:
                                hbg.add_edge(centNode,nnode)
                                ningraph.append(N)
            
        else:
            hi = bbn+1
            h = top.atom(hi)
            if h.name!='H':
                #sanity check for hydrogen & bonds
                print "At {0}, did not find a bonded hydrogen.\n".format(bbn)
                return -1
            else:
                for wn in wns:
                    hbool = hbond(frame,bbn,wn,hi,hbang)
                    if hbool:
                        #we have found a hydrogen bond
                        #we add the water to the graph, and we add an edge between it and the central node
                        wnode = HBNode('W',wn)
                        
                        hbg.add_node(wnode)
                        wingraph.append(wn)
                        hbg.add_edge(centNode,wnode)
                      
                        #^^THIS PART IS SANITY-CHECK APPROVED (picked up all and no more of the Hbonds that VMD did in one particular frame--should keep checking as I go on with different frames)
                for N in nns:
                    #(ii) if SC ns are an acceptor, the SC nitrogens - either way add SC nitrogens to graph
                    #the SC N indices are in Ninds
                    nnode = HBNode('SCN',N)
                     
                    if naccbool:
                        hbool = hbond(frame,bbn,N,hi,hbang)
                        if hbool:
                            hbg.add_edge(centNode,nnode)
                            ningraph.append(N)
        return 1
                            
def bboparse(frame,hbg,bbos,naccbool,woxsel,hbcut,Ninds,Cinds,wingraph,ningraph,cingraph,hbang,centNode):
    #make this its own function so we can look at this whole thing more easily
    #this is the part that does the backbone oxygens & their possible h bonds
    #check whether there are bonds with
    
    
    for bbo in bbos:
        #(i) water as a donor - if so, add water to graph
        wns = md.compute_neighbors(frame,hbcut,np.array([bbo]),woxsel)[0] #candidate oxygen donors
        #I ASSUME AT THIS POINT THAT THE NEXT TWO ATOMS IN THE TOPOLOGY FROM THE O ARE THE H1 AND THE H2
        #IF THEY AREN'T, WE BETTER THROW AN ERROR
        nns = md.compute_neighbors(frame,hbcut,np.array([bbo]),np.array(Ninds))[0]
        cns = md.compute_neighbors(frame,hbcut,np.array([bbo]),np.array(Cinds))[0]
        for wn in wns:
            for j in range(2):
                hi = wn+j+1
                h = top.atom(hi)
                if h.name!='H'+str(j+1):
                    print "At {0} (water), did not find a bonded hydrogen.\n".format(wn)
                    return -1
                hbool = hbond(frame,wn,bbo,hi,hbang)
                if hbool:
                    #we have found a hydrogen bond
                    #we add the water to the graph, and we add an edge between it and the central node
                    wnode = HBNode('W',wn)
                    
                    hbg.add_node(wnode)
                    wingraph.append(wn)
                    hbg.add_edge(centNode,wnode)
            
  
                  
        if not naccbool:
            #(ii) if SC ns are a donor, the SC nitrogens 
            #the SC N indices are in Ninds
            for N in nns:
                nnode = HBNode('SCN',N)
                hi = N+1
                h = top.atom(hi)
                if h.name!='H':
                    print "At {0} (SCN), did not find a bonded hydrogen.\n".format(N)
                    return -1
                hbool = hbond(frame,N,bbo,hi,hbang)
                if hbool:
                    hbg.add_edge(centNode,nnode)
                    ningraph.append(N)
        for C in cns:
            #(iii) SC carbons
            cnode = HBNode('SCC',C)
            hi = C+1
            h = top.atom(hi)
            if h.name!='H':
                print "At {0} (SCC), did not find a bonded hydrogen.\n".format(C)
                return -1
            hbool = hbond(frame,C,bbo,hi,hbang)
            if hbool:
                hbg.add_edge(centNode,cnode)
                cingraph.append(C)
        return 1
                            
def hbgraph(frame,Cinds,Ninds,ndxfile,naccbool,hbcut=0.35,hbang=(120.0*np.pi/180.0)):
    #takes a frame of a trajectory and turns it into a graph
    #where we have a protein, the first node of the graph is a lump of all bb nitrogens and oxygens
    #then the SC C donors and N donors (triazole) or acceptors (triazolium) are also in the graph
    #as are all waters that are h-bond-connected to the network 
    #Cinds are the locations of the side chain carbons
    #Ninds are the locations of the side chain nitrogens
    #if naccbool = true, the SC nitrogens are acceptors; else they are donors
    top = frame.topology
    #instantiate graph and add the central node
    hbg = nx.Graph()
    centNode = HBNode('B',-1)
    hbg.add_node(centNode) 
    bbsel = groupFromNdx(ndxfile,'[ Backbone ]')
    nsall = top.select("name N")
    osall = top.select("not water and name O")
    woxsel = top.select("water and name O")
    #get backbone nitrogens+associated hydrogens (these are in BB selection, named N)
    #end residue nitrogens have THREE associated hydrogens that might be involved in bonding!
    bbns = list(set(bbsel) & set(nsall))
    #get backbone oxygens (these are in BB selection, named O)
    bbos = list(set(bbsel) & set(osall))
    wingraph = [] #for debugging only
    ningraph = [] #for debugging only
    cingraph = [] #for debugging only
    #then check whether there are bonds with (*)
    for N in Ninds:
        nnode = HBNode('SCN',N)
        hbg.add_node(nnode)
    for C in Cinds:
        cnode = HBNode('SCC',C)
        hbg.add_node(cnode)
    herr = bbnparse(frame,hbg,bbns,naccbool,woxsel,hbcut,Ninds,wingraph,ningraph,hbang,centNode)
    if herr==-1:
        return []
    
    herr  = bboparse(frame,hbg,bbos,naccbool,woxsel,hbcut,Ninds,Cinds,wingraph,ningraph,cingraph,hbang,centNode)
    if herr==-1:
        return []
    #next check if there are bonds with SC carbons to water - add any waters to graph
    
    #next check if there are bonds with SC nitrogens to water - add any waters to graph
    
    #if SC ns are acceptors, find bonds with SC carbons/SC nitrogens
    
    #finally go over all waters in graph and find bonds to other waters.  Waters that are not in graph get added and also checked
    #maybe use a list + pop structure to track?
    
    return (hbg,wingraph,ningraph,cingraph)
        
if __name__ == "__main__":
    #we want to load the trajectory in, get (oxygen) of water, side chains, and backbone
    #then we can check the distance to various things
    folder = '/home/rachael/JJsims/hbonding/triazole/6x_md_CAT/'
    #trajfile = folder+'md.trr'
        
    trajfile = folder+'md_whole0.gro'    
    topfile = folder+'after_md_PRIOR.gro'
    ptopfile = folder+'after_md_PRIOR_protein_fit.gro'
    ptoplines = 246 #lines in ptopfile
    ndxfile = folder+'index.ndx'
    cutoff = 0.7 #trr file is in nm, ps    
    itpfile = folder+'PTPLG.top'
    outfname = folder+'md_whole0.gro'
    Cd = np.array([19,43,67,91,115,139,163,187,211,236]) #triazole carbon donor indices on side chain
    Na = np.array([17,41,65,89,113,137,161,185,209,234]) #triazole nitrogen acceptors on side chain
    #Cd = np.array([25,50,75,100,125,150,175,200,225,251])   #triazolium carbon donors
    #Na = np.array([17,42,67,92,117,142,167,192,217,243]) #triazolium nitrogen donors
    
    
    traj = md.load(trajfile,top=topfile)
   
    top = traj.topology

    #addBonds(itpfile,top)    
    #ptop = groTop(ptopfile,ptoplines)
    #minds = minDistsT(traj,ndxfile,'mindistW-BB.dat')
    #make trajectory of whole + water shell (this isn't super efficient right now)
    (hbnn,win,nin,cin) = hbgraph(traj[0],Cd,Na,ndxfile,False)
    print win    
    print nin
    print cin
    #moveShell(traj[0],top,ptop,cutoff,ndxfile,outfname,'w')
    
    #for f in range(1,traj.n_frames):
     #   print f
      #  moveShell(traj[f],top,ptop,cutoff,ndxfile,outfname,'a')
