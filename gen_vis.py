import glob, os
import numpy as np
from os import listdir
from os.path import isfile, join

vis_root = "vis"

pattern = os.environ["MODEL"]
print "working on %s" % pattern
# nets = [n for n in listdir(vis_root) if not isfile(join(vis_root, n))]
# print nets
vis_pattern = os.path.join(vis_root, pattern)

nets = sorted(glob.glob(vis_pattern))
print nets

do_unsup = True
do_merge = False

for net in nets:
    net = net[4:]
    print net

    if net.find("D") == -1:
        do_depth = False
    else:
        print "doing depth"
        do_depth = True
    if net.find("surf") == -1:
        do_surf = False
    else:
        print "doing surf"
        do_surf = True
        
    if net.find("M") == -1:
        do_mask = False
    else:
        print "doing mask"
        do_mask = True
    if net.find("F") == -1:
        do_flow = False
    else:
        print "doing flow"
        do_flow = True
    if net.find("_O") == -1:
        do_odo = False
    else:
        print "doing odo"
        do_odo = True
    if net.find("P") == -1:
        do_pose = False
    else:
        print "doing pose"
        do_pose = True
    if net.find("_C") == -1:
        do_com = False
    else:
        print "doing com"
        do_com = True
    if net.find("_S") == -1:
        print "doing seg"
        do_seg = False
    else:
        do_seg = True
        
    if net.find("backward") == -1:
        do_backward = False
    else:
        do_backward = True

    img_path = "vis/%s" % net
    display = 1
    
    f = open("vis_%s.html" % net, "w")

    with open("vis_start.html") as fs:
        content = fs.readlines()
        content = [x.strip() for x in content]
        for l in content:
            f.write(l)
            f.write('\n')

    pattern = os.path.join(img_path, "*_i1i2.gif")
    pngs = sorted(glob.glob(pattern),reverse=True)
    nPngs = len(pngs)
    if nPngs > 100:
        nPngs = 100
    
    print nPngs, "pngs"

    count = 12
    
    # if do_flow:
    #     count += 3
    #     if do_backward:
    #         count += 2
    # if do_depth:
    #     count += 2
    # if do_odo:
    #     count += 4
    #     if do_backward:
    #         count += 2

    p=str(int(100./count))

    if nPngs==0:
        f.write('<td width="100%"><div class="label">Couldn\'t find any pngs in '+img_path+'</div></td>')
    else:
        f.write('<tr>\n')

        if do_flow or do_odo or do_pose:
            f.write('<td width="'+p+'%"><div class="top">i1i2</div></td>')
        elif do_mask or do_depth or do_seg:
            f.write('<td width="'+p+'%"><div class="top">i1</div></td>')
        
        if do_seg:
            f.write('<td width="'+p+'%"><div class="top">s1_e</div></td>')
            f.write('<td width="'+p+'%"><div class="top">s1_g</div></td>')
        if do_mask:
            if not (do_pose or do_com):
                f.write('<td width="'+p+'%"><div class="top">m1_e</div></td>')
                f.write('<td width="'+p+'%"><div class="top">m1_g</div></td>')
            f.write('<td width="'+p+'%"><div class="top">m1_m</div></td>')
            
        if do_flow:
            if do_odo:
                f.write('<td width="'+p+'%"><div class="top">f12_e</div></td>')
                f.write('<td width="'+p+'%"><div class="top">fi1i1</div></td>')
            else:
                f.write('<td width="'+p+'%"><div class="top">f12_e</div></td>')
                # f.write('<td width="'+p+'%"><div class="top">f12_g</div></td>')
                f.write('<td width="'+p+'%"><div class="top">fi1i1</div></td>')
                if do_backward:
                    f.write('<td width="'+p+'%"><div class="top">f21_e</div></td>')
                    f.write('<td width="'+p+'%"><div class="top">fi2i2</div></td>')
        if do_depth:
            if do_surf:
                # f.write('<td width="'+p+'%"><div class="top">sm_e</div></td>')
                f.write('<td width="'+p+'%"><div class="top">sn_e</div></td>')
                f.write('<td width="'+p+'%"><div class="top">sn_g</div></td>')
            if do_odo:
                f.write('<td width="'+p+'%"><div class="top">d1_e</div></td>')
                # f.write('<td width="'+p+'%"><div class="top">d1_g</div></td>')
            elif not (do_pose or do_com):
                if not do_unsup:
                    f.write('<td width="'+p+'%"><div class="top">v1</div></td>')
                f.write('<td width="'+p+'%"><div class="top">d1_e</div></td>')
                f.write('<td width="'+p+'%"><div class="top">d1_g</div></td>')
                if do_merge:
                    f.write('<td width="'+p+'%"><div class="top">d1_m</div></td>')
            else:
                f.write('<td width="'+p+'%"><div class="top">d1_m</div></td>')
        if do_odo:
            # f.write('<td width="'+p+'%"><div class="top">m1_g</div></td>')
            f.write('<td width="'+p+'%"><div class="top">o12_e</div></td>')
            f.write('<td width="'+p+'%"><div class="top">o12_g</div></td>')
            f.write('<td width="'+p+'%"><div class="top">oi1i1</div></td>')
            f.write('<td width="'+p+'%"><div class="top">oi1i1_g</div></td>')
            if do_backward:
                f.write('<td width="'+p+'%"><div class="top">o21_e</div></td>')
                f.write('<td width="'+p+'%"><div class="top">o21_g</div></td>')
        if do_pose:
            f.write('<td width="'+p+'%"><div class="top">p12_e</div></td>')
            f.write('<td width="'+p+'%"><div class="top">p12_e2</div></td>')
            f.write('<td width="'+p+'%"><div class="top">pi1i1</div></td>')
            f.write('<td width="'+p+'%"><div class="top">pi1i1_2</div></td>')
        if do_com:
            f.write('<td width="'+p+'%"><div class="top">c12_e</div></td>')
            # f.write('<td width="'+p+'%"><div class="top">c12_e2</div></td>')
            f.write('<td width="'+p+'%"><div class="top">ci1i1</div></td>')
            f.write('<td width="'+p+'%"><div class="top">ti1i2</div></td>')
            # f.write('<td width="'+p+'%"><div class="top">ci1i1_2</div></td>')
            
        # f.write('<td width="'+p+'%"><div class="top">m1_e</div></td>')
        # f.write('<td width="'+p+'%"><div class="top">i1i1_o</div></td>')
        
        # f.write('<td width="'+p+'%"><div class="top">f21_o</div></td>')
        # f.write('<td width="'+p+'%"><div class="top">z1t</div></td>')
        # f.write('<td width="'+p+'%"><div class="top">z2t</div></td>')
        # f.write('<td width="'+p+'%"><div class="top">zdz</div></td>')
        # f.write('<td width="'+p+'%"><div class="top">mez</div></td>')
        f.write('</tr>\n')

    for i in range(0,nPngs):
        if np.mod(i, display) == 0:
            flow = pngs[i]
            lp = len(img_path)
            num = flow[lp+1:-9]
            print num
            i1 = img_path+'/'+num+'_i1_g.png'
            i1i2 = img_path+'/'+num+'_i1i2.gif'
            fi1i1 = img_path+'/'+num+'_fi1i1.gif'
            fi2i2 = img_path+'/'+num+'_fi2i2.gif'
            i1i1_o = img_path+'/'+num+'_i1i1_o.gif'
            i1i1m = img_path+'/'+num+'_i1i1m.gif'
            i2i2 = img_path+'/'+num+'_i2i2.gif'
            i2i2_o = img_path+'/'+num+'_i2i2_o.gif'

            s1_e = img_path+'/'+num+'_s1_e.png'
            s1_g = img_path+'/'+num+'_s1_g.png'
            
            f12_e = img_path+'/'+num+'_f12_e.png'
            f12_g = img_path+'/'+num+'_f12_g.png'
            f21_e = img_path+'/'+num+'_f21_e.png'

            o12_e = img_path+'/'+num+'_o12_e.png'
            o12_g = img_path+'/'+num+'_o12_g.png'
            o21_e = img_path+'/'+num+'_o21_e.png'
            o21_g = img_path+'/'+num+'_o21_g.png'
            oi1i1 = img_path+'/'+num+'_oi1i1.gif'
            oi1i1_g = img_path+'/'+num+'_oi1i1_g.gif'
            oi2i2 = img_path+'/'+num+'_oi2i2.gif'

            p12_e = img_path+'/'+num+'_p12_e.png'
            p12_e2 = img_path+'/'+num+'_p12_e2.png'
            pi1i1 = img_path+'/'+num+'_pi1i1.gif'
            pi1i1_2 = img_path+'/'+num+'_pi1i1_2.gif'

            c12_e = img_path+'/'+num+'_c12_e.png'
            c12_e2 = img_path+'/'+num+'_c12_e2.png'
            ci1i1 = img_path+'/'+num+'_ci1i1.gif'
            ci1i1_2 = img_path+'/'+num+'_ci1i1_2.gif'
            ti1i2 = img_path+'/'+num+'_ti1i2.gif'

            m1_e = img_path+'/'+num+'_m1_e.png'
            m1_m = img_path+'/'+num+'_m1_m.png'
            m1_g = img_path+'/'+num+'_m1_g.png'
            
            d1_e = img_path+'/'+num+'_d1_e.png'
            d1_m = img_path+'/'+num+'_d1_m.png'
            d1_g = img_path+'/'+num+'_d1_g.png'
            v1 = img_path+'/'+num+'_v1.png'
            sm_e = img_path+'/'+num+'_sm_e.png'
            sn_e = img_path+'/'+num+'_sn_e.png'
            sn_g = img_path+'/'+num+'_sn_g.png'
            
            z1t = img_path+'/'+num+'_z1t.png'
            z2t = img_path+'/'+num+'_z2t.png'
            zdz = img_path+'/'+num+'_zdz.png'
            mez = img_path+'/'+num+'_mez.png'

            f.write('<tr>\n')

            if do_flow or do_odo or do_pose or do_com:
                f.write('<td width="'+p+'%"><div class="label">'+num+'</div><img src="'+i1i2+'"></td>')
                #elif do_mask or do_depth:
            else:
                f.write('<td width="'+p+'%"><div class="label">'+num+'</div><img src="'+i1+'"></td>')

            if do_seg:
                f.write('<td width="'+p+'%"><img src="'+s1_e+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+s1_g+'"></td>')
                
            if do_mask:
                if not (do_pose or do_com):
                    f.write('<td width="'+p+'%"><img src="'+m1_e+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+m1_g+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+m1_m+'"></td>')
                
            if do_flow:
                if do_odo:
                    f.write('<td width="'+p+'%"><img src="'+f12_e+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+fi1i1+'"></td>')
                else:
                    f.write('<td width="'+p+'%"><img src="'+f12_e+'"></td>')
                    # f.write('<td width="'+p+'%"><img src="'+f12_g+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+fi1i1+'"></td>')
                    if do_backward:
                        f.write('<td width="'+p+'%"><img src="'+f21_e+'"></td>')
                        f.write('<td width="'+p+'%"><img src="'+fi2i2+'"></td>')
            if do_depth:
                if do_surf:
                    # f.write('<td width="'+p+'%"><img src="'+sm_e+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+sn_e+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+sn_g+'"></td>')
                if do_odo:
                    f.write('<td width="'+p+'%"><img src="'+d1_e+'"></td>')
                    # f.write('<td width="'+p+'%"><img src="'+d1_g+'"></td>')
                elif not (do_pose or do_com):
                    if not do_unsup:
                        f.write('<td width="'+p+'%"><img src="'+v1+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+d1_e+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+d1_g+'"></td>')
                    if do_merge:
                        f.write('<td width="'+p+'%"><img src="'+d1_m+'"></td>')
                else:
                    f.write('<td width="'+p+'%"><img src="'+d1_m+'"></td>')

            if do_odo:
                # f.write('<td width="'+p+'%"><img src="'+m1_g+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+o12_e+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+o12_g+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+oi1i1+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+oi1i1_g+'"></td>')
                if do_backward:
                    f.write('<td width="'+p+'%"><img src="'+o21_e+'"></td>')
                    f.write('<td width="'+p+'%"><img src="'+o21_g+'"></td>')
            if do_pose:
                f.write('<td width="'+p+'%"><img src="'+p12_e+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+p12_e2+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+pi1i1+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+pi1i1_2+'"></td>')
            # f.write('<td width="'+p+'%"><img src="'+m1_e+'"></td>')
            # f.write('<td width="'+p+'%"><img src="'+i1i1_o+'"></td>')

            if do_com:
                f.write('<td width="'+p+'%"><img src="'+c12_e+'"></td>')
                # f.write('<td width="'+p+'%"><img src="'+c12_e2+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+ci1i1+'"></td>')
                f.write('<td width="'+p+'%"><img src="'+ti1i2+'"></td>')
                # f.write('<td width="'+p+'%"><img src="'+ci1i1_2+'"></td>')

            
            # f.write('<td width="'+p+'%"><img src="'+f21_o+'"></td>')
            # f.write('<td width="'+p+'%"><img src="'+z1t+'"></td>')
            # f.write('<td width="'+p+'%"><img src="'+z2t+'"></td>')
            # f.write('<td width="'+p+'%"><img src="'+zdz+'"></td>')
            # f.write('<td width="'+p+'%"><img src="'+mez+'"></td>')

            f.write('</tr>\n')

    with open("vis_end.html") as fs:
        content = fs.readlines()
        content = [x.strip() for x in content]
        for l in content:
            f.write(l)
            f.write('\n')

    f.close()
