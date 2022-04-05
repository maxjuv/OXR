from configuration import *

from select_mice_cata_Malo import *

#
def compute_all():
    for group in groups :
        print(group)
        if len(get_mice(group)) != 0:
            precompute_sleep_state_by_epoch(group)
            sleep_state_statistics(group)
            sleep_bouts(group)
            inter_REM_interval(group)
    plot_cata_number_accross_time()
    plot_tdw_before_cata()


def precompute_sleep_state_by_epoch(group):

    mice = get_mice(group)
    print(mice)

    days = [ 'b1', 'b2', 'sd', 'r1' ]

    all_mice_all_days = {}
    numbers = ['1', '2', '3', '4' ,'5' ,'6']
    letters = ['w', 'n', 'r', 'w', 'n', 'r']

    # for mouse in mice :
    #     data = []
    #     for day in days :
    #         print(data_dir  + group+ "/" + mouse + 'DCR'+day + ".txt")
    #         data_per_day = np.loadtxt(data_dir  +"/Scoring/"+ group+ "/" + mouse + 'DCR'+day + ".txt", dtype = str)
    #         for number, letter in zip (numbers, letters):
    #             data_per_day = np.where(data_per_day == number, letter, data_per_day)
    #         data.append(data_per_day)
    dirname = precompute_dir + '/tdw_score/'
    for mouse in mice :
        print(mouse)
        ds_tdw = xr.open_dataset(dirname + 'tdw_score_{}.nc'.format(mouse))
        new_score = ds_tdw['new_score'].values
        for number, letter in zip (numbers, letters):
            new_score = np.where(new_score == number, letter, new_score)
        all_mice_all_days[mouse] = new_score
    all_mice_all_days = pd.DataFrame.from_dict(all_mice_all_days)

    # all_mice_all_days = all_mice_all_days.T

    print('find w, n, r')
    all_mice_w_epoch = (all_mice_all_days == 'w') | (all_mice_all_days == 't')
    all_mice_tdw_epoch = all_mice_all_days == 't'
    all_mice_n_epoch = all_mice_all_days == 'n'
    all_mice_r_epoch = all_mice_all_days == 'r'
    all_mice_cata_epoch = all_mice_all_days == 'a'

    print('create Dataset')
    path = precompute_dir + '/' + group + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # ds.to_netcdf(path + 'summed_sleep_by_epoch.nc', mode='w')
    epochs = np.arange(all_mice_all_days.shape[0])
    # print(epochs)
    # print(type(mice))
    coords = {'mice':mice, 'epochs':epochs }
    ds = xr.Dataset(coords = coords)
    # print(all_mice_w_epoch)

    ds['wake_all_mice'] = xr.DataArray(all_mice_w_epoch*1, dims = ['epochs','mice'])
    ds['nrem_all_mice'] = xr.DataArray(all_mice_n_epoch*1, dims = ['epochs','mice'])
    ds['cata_all_mice'] = xr.DataArray(all_mice_cata_epoch*1, dims = ['epochs','mice'])
    ds['tdw_all_mice'] = xr.DataArray(all_mice_tdw_epoch*1, dims = ['epochs','mice'])
    ds['rem_all_mice'] = xr.DataArray(all_mice_r_epoch*1, dims = ['epochs','mice'])
    # print(ds)
    # exit()

    print('save')
    ds.to_netcdf(path + 'sleep_by_epoch.nc', mode='w')

    #
def sleep_state_statistics( group):
    path = precompute_dir+'/'+ group + '/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)
    level = [0, 1]
    # print(ds)
    all_mice_w_epoch = ds['wake_all_mice'].to_pandas()
    all_mice_n_epoch = ds['nrem_all_mice'].to_pandas()
    all_mice_t_epoch = ds['tdw_all_mice'].to_pandas()
    all_mice_r_epoch = ds['rem_all_mice'].to_pandas()
    all_mice_cata_epoch = ds['cata_all_mice'].to_pandas()

    number_of_epochs = all_mice_w_epoch.shape[0]
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    hours = np.arange(0, int(duration_in_hours),time)
    window = int(time * 3600 / epoch_duration)

    fake = np.zeros(number_of_epochs)

    all_mice_w_by_time = []
    all_mice_t_by_time = []
    all_mice_cata_by_time = []
    all_mice_r_by_time = []
    all_mice_n_by_time = []
    for hour in hours :
        i1, i2 = int(hour*window), int((hour+1)*window)
        # tarace[hour] = all_mice_w_epoch[i1:i2].sum(axis =0)
        all_mice_w_by_time.append(all_mice_w_epoch[i1:i2].sum(axis =0)*4/60)
        all_mice_t_by_time.append(all_mice_t_epoch[i1:i2].sum(axis =0)*4/60)
        all_mice_r_by_time.append(all_mice_r_epoch[i1:i2].sum(axis =0)*4/60)
        all_mice_n_by_time.append(all_mice_n_epoch[i1:i2].sum(axis =0)*4/60)
        all_mice_cata_by_time.append(all_mice_cata_epoch[i1:i2].sum(axis =0)*4/60)
        # print(all_mice_cata_epoch[i1:i2].sum(axis =0)*4/60)
    # print(len(all_mice_cata_by_time))
    all_mice_cata_by_time = pd.concat(all_mice_cata_by_time, axis = 1)
    all_mice_w_by_time = pd.concat(all_mice_w_by_time, axis = 1)
    all_mice_t_by_time = pd.concat(all_mice_t_by_time, axis = 1)
    all_mice_r_by_time = pd.concat(all_mice_r_by_time, axis = 1)
    all_mice_n_by_time = pd.concat(all_mice_n_by_time, axis = 1)

    if not os.path.exists(excel_dir+  '/'+ group + '/time_dynamics/'):
        os.makedirs(excel_dir+ '/'+ group + '/time_dynamics/')

    all_mice_w_by_time.to_excel(excel_dir+ '/'+ group + '/time_dynamics/'+ '/wake_event_by_mouse_by_hour.xlsx')
    all_mice_t_by_time.to_excel(excel_dir+ '/'+ group + '/time_dynamics/'+ '/tdw_event_by_mouse_by_hour.xlsx')
    all_mice_n_by_time.to_excel(excel_dir+ '/'+ group +  '/time_dynamics/'+ '/NREM_event_by_mouse_by_hour.xlsx')
    all_mice_r_by_time.to_excel(excel_dir+ '/'+ group +  '/time_dynamics/'+ '/REM_event_by_mouse_by_hour.xlsx')
    all_mice_cata_by_time.to_excel(excel_dir+ '/'+ group +  '/time_dynamics/'+ '/cata_event_by_mouse_by_hour.xlsx')




def plot_sleep_state_accross_time(group):
    # data_dir = 'C:/Users/maxime.juventin/Desktop/scripts_ML/data/'
    path = precompute_dir + '/' + group + '/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)

    print(ds)
    number_of_epochs = ds['wake_all_mice'].shape[0]
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    wake = ds['wake_all_mice'].values
    tdw = ds['tdw_all_mice'].values
    nrem = ds['nrem_all_mice'].values
    rem = ds['rem_all_mice'].values
    cata = ds['cata_all_mice'].values

    time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    window = int(time * 3600 / epoch_duration)
    wake_by_time = np.zeros(int(duration_in_hours/time))
    tdw_by_time = np.zeros(int(duration_in_hours/time))
    rem_by_time = np.zeros(int(duration_in_hours/time))
    cata_by_time = np.zeros(int(duration_in_hours/time))
    nrem_by_time = np.zeros(int(duration_in_hours/time))
    for h, i in enumerate(np.arange(0, number_of_epochs, window)):
        wake_by_time[h] = wake[i : i+window].sum()
        tdw_by_time[h] = wake[i : i+window].sum()
        cata_by_time[h] = cata[i : i+window].sum()
        rem_by_time[h] = rem[i : i+window].sum()
        nrem_by_time[h] = nrem[i : i+window].sum()
    print(wake_by_time.shape)
    fig, ax = plt.subplots(nrows=5)
    times = np.arange(0, duration_in_hours,time)
    ax[0].plot(times, wake_by_time*epoch_duration/60, color = 'black', label = 'wake')
    ax[1].plot(times, tdw_by_time*epoch_duration/60, color = 'red', label='rem')
    ax[2].plot(times, rem_by_time*epoch_duration/60, color = 'red', label='rem')
    ax[3].plot(times, nrem_by_time*epoch_duration/60, color = 'blue', label = 'nrem')
    ax[4].plot(times, cata_by_time*epoch_duration/60, color = 'blue', label = 'cata')
    for i,j in zip([0,28-time,76-time], [4-time,52-time,100-time]):
        ax[0].axvspan(i,j , color = 'black', alpha = .3)
        ax[1].axvspan(i,j , color = 'black', alpha = .3)
        ax[2].axvspan(i,j , color = 'black', alpha = .3)
        ax[3].axvspan(i,j , color = 'black', alpha = .3)
        ax[4].axvspan(i,j , color = 'black', alpha = .3)
    ax[0].set_title('Sleep state amount for '+ group + ' , sum per ' + str(time) + ' hours')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    plt.show()

def wake_time_before_cata():
    path = precompute_dir + '/Ox1r_Ox2r_dd/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)

    # Get cata event, keep 1st position
    # Verify always wake before
    # count backward wake size
    # give halday and by hour


    # print(ds)
    number_of_epochs = ds['wake_all_mice'].shape[0]
    wake = ds['wake_all_mice'].to_pandas()
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    # print(ds['cata_all_mice'].to_pandas())
    cata = ds['cata_all_mice'].to_pandas()
    time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    window = int(time * 3600 / epoch_duration)
    cata_by_time = np.zeros(int(duration_in_hours/time))
    cata_count_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    cata_duration_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    mice = cata.columns.to_list()
    for mouse in mice :
        # print(mouse)
        for h, hour in enumerate(np.arange(0, number_of_epochs, window)):
            # print('hour = ',h)
            subs = cata[mouse][hour : hour+window].values
            sub_wake = wake[mouse][hour : hour+window].values

            if subs.sum() == 0 :
                # print( 'yep   ', h)
                cata_count_by_hour.at[h, mouse]=0
            else :
                counter = 0
                cata_duration = []
                cata_number = []
                positions =[]
                wake_pre_cata_duration = []
                for pos,i in enumerate(subs):
                    if i ==0 and counter !=0:
                        cata_number.append(counter)
                        if len(positions)>1:
                            ini = min(positions)
                            end =  max(positions)
                        elif len(positions)==1 :
                            ini = positions[0]
                            end =  ini +1
                        cata_duration.append([ini,end])

                        if sub_wake[ini-1]!= 1 and subs[ini-1]!= 1:
                            print('##### ENCODING ERROR ##### cataplexy during sleep')
                        else :
                            inverted_wake_for_counting = sub_wake[:ini]
                            inverted_wake_for_counting = inverted_wake_for_counting[::-1]
                            wake_count = 0
                            for w in inverted_wake_for_counting:
                                if w == 0:
                                    break
                                else :
                                    wake_count +=1
                        # fig, ax = plt.subplots()
                        # ax.plot(subs)
                        # ax.plot(sub_wake)
                        # # print(positions)
                        # print(wake_count)
                        # print(end-ini)
                        cata_duration.append(ini-end)
                        wake_pre_cata_duration.append(wake_count)
                        # plt.show()
                        positions = []
                        counter = 0
                    if i ==1:
                        counter +=1
                        positions.append(pos)
                # exit()

                # cata_number.append(counter) #### !!!! Add last overlapping event would count twice the same event : end of hour h and begin of hour h+1
                cata_number=np.array(cata_number)
                mask = cata_number!=0
                cata_number = cata_number[mask]
                cata_duration = np.mean(cata_number*4/60)
                cata_count_by_hour.at[h, mouse]= cata_number.size
                cata_duration_by_hour.at[h, mouse]= cata_duration
        exit()
    # dirname = excel_dir + '/Ox1r_Ox2r_d/d/cataplexy_count/'
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # filename =dirname+'by_hour_cataplexy_count.xlsx'
    # cata_count_by_hour.to_excel(filename)
    #
    # dirname = excel_dir + '/Ox1r_Ox2r_d/d/cataplexy_duration/'
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # filename =dirname+'by_hour_cataplexy_duration.xlsx'
    # cata_duration_by_hour.to_excel(filename)

    time = int(12) # window to look at in HOUR. CAUTION must be a multiple of 100
    window = int(time * 3600 / epoch_duration)
    cata_by_time = np.zeros(int(duration_in_hours/time))
    # halfday_dict = {0:'bl1_light',1:'bl1_dark',
    #                 2:'bl2_light',3:'bl2_dark'
    #                 4:'sd_light',5:'sd_dark'
    #                 6:'r1_light',7:'r1_dark'}
    cata_count_by_halfday = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    cata_duration_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    for mouse in mice :
        # print(mouse)
        for h, i in enumerate(np.arange(0, number_of_epochs, window)):
            # print('hour = ',h)
            subs = cata[mouse][i : i+window]
            if subs.sum() == 0 :
                # print( 'yep   ', h)
                cata_count_by_halfday.at[h, mouse]=0
            else :
                counter = 0
                cata_number = []
                for i in subs:
                    if i ==0:
                        cata_number.append(counter)
                        counter = 0
                    if i ==1:
                        counter +=1
                # cata_number.append(counter) #### !!!! Add last overlapping event would count twice the same event : end of hour h and begin of hour h+1
                cata_number=np.array(cata_number)
                mask = cata_number!=0
                cata_number = cata_number[mask]
                cata_duration = np.mean(cata_number*4/60)
                cata_duration_by_hour.at[h, mouse]= cata_duration
                cata_count_by_halfday.at[h, mouse]= cata_number.size
    # dirname = excel_dir + '/Ox1r_Ox2r_d/d/cataplexy_count/'
    # filename =dirname+'by_halfday_cataplexy_count.xlsx'
    # cata_count_by_halfday.to_excel(filename)
    #
    # dirname = excel_dir + '/Ox1r_Ox2r_d/d/cataplexy_duration/'
    # filename =dirname+'by_halfday_cataplexy_duration.xlsx'
    # cata_duration_by_hour.to_excel(filename)

def plot_cata_number_accross_time():
    # data_dir = 'C:/Users/maxime.juventin/Desktop/scripts_ML/data/'
    path = precompute_dir + '/Ox1r_Ox2r_dd/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)

    # print(ds)
    number_of_epochs = ds['wake_all_mice'].shape[0]
    wake = ds['wake_all_mice'].to_pandas()
    rem = ds['rem_all_mice'].to_pandas()
    nrem = ds['nrem_all_mice'].to_pandas()
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    # print(ds['cata_all_mice'].to_pandas())
    cata = ds['cata_all_mice'].to_pandas()
    time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    window = int(time * 3600 / epoch_duration)
    cata_by_time = np.zeros(int(duration_in_hours/time))

    # empty_matrix = np.empty((cata_by_time.size,cata.shape[1]))
    # empty_matrix[:]=np.nan
    cata_count_by_hour = pd.DataFrame(np.zeros((cata_by_time.size,cata.shape[1])), index = np.arange(cata_by_time.size), columns = cata.columns.to_list())
    cata_duration_by_hour = pd.DataFrame(np.zeros((cata_by_time.size,cata.shape[1])), index = np.arange(cata_by_time.size), columns = cata.columns.to_list())
    wake_pre_cata_duration_by_hour = pd.DataFrame(np.zeros((cata_by_time.size,cata.shape[1])), index = np.arange(cata_by_time.size), columns = cata.columns.to_list())
    mice = cata.columns.to_list()

    for mouse in mice :
        # print(mouse)
        for h, hour in enumerate(np.arange(0, number_of_epochs, window)):
            # print('hour = ',h)
            subs = cata[mouse][hour : hour+window].values
            sub_wake = wake[mouse][hour : hour+window].values
            positions =[]
            wake_pre_cata_duration = []

            if subs.sum() == 0 :
                # print( 'yep   ', h)

                cata_count_by_hour.at[h, mouse]=0
                cata_duration_by_hour.at[h, mouse]=0
            else :
                counter = 0
                cata_event_size = []
                for pos,i in enumerate(subs):
                    if i ==0 and counter !=0:
                        cata_event_size.append(counter)
                        counter = 0
                        ini = positions[0]
                        if ini !=0 :
                            # print(ini)
                            if sub_wake[ini-1]!= 1 and subs[ini-1]!= 1:
                                print('##### ENCODING ERROR ##### cataplexy during sleep')
                                sub_rem = rem[mouse][hour : hour+window].values
                                if  sub_rem[ini-1] ==1:
                                    print('REM')
                                sub_nrem = nrem[mouse][hour : hour+window].values
                                if  sub_nrem[ini-1] ==1:
                                    print('NREM')
                            else :
                                inverted_wake_for_counting = sub_wake[:ini]
                                inverted_wake_for_counting = inverted_wake_for_counting[::-1]
                                wake_count = 0
                                for w in inverted_wake_for_counting:
                                    if w == 0:
                                        break
                                    else :
                                        wake_count +=1
                                wake_pre_cata_duration.append(wake_count)
                    if i ==1:
                        counter +=1
                        positions.append(pos)
                # cata_number.append(counter) #### !!!! Add last overlapping event would count twice the same event : end of hour h and begin of hour h+1
                cata_event_size=np.array(cata_event_size)
                mask = cata_event_size!=0
                cata_event_size = cata_event_size[mask]
                wake_pre_cata_duration = np.mean(wake_pre_cata_duration)*4/60
                cata_duration = np.mean(cata_event_size*4/60)

                cata_count_by_hour.at[h, mouse]= cata_event_size.size
                cata_duration_by_hour.at[h, mouse]= cata_duration
                wake_pre_cata_duration_by_hour.at[h, mouse]= wake_pre_cata_duration

    dirname = excel_dir + '/Ox1r_Ox2r_dd/cataplexy_count/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename =dirname+'by_hour_cataplexy_count.xlsx'
    cata_count_by_hour.to_excel(filename)

    dirname = excel_dir + '/Ox1r_Ox2r_dd/cataplexy_duration/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename =dirname+'by_hour_cataplexy_duration.xlsx'
    cata_duration_by_hour.to_excel(filename)

    dirname = excel_dir + '/Ox1r_Ox2r_dd/wake_pre_cataplexy_duration/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename =dirname+'by_hour_cataplexy_duration.xlsx'
    wake_pre_cata_duration_by_hour.to_excel(filename)

    time = int(12) # window to look at in HOUR. CAUTION must be a multiple of 100
    window = int(time * 3600 / epoch_duration)
    cata_by_time = np.zeros(int(duration_in_hours/time))
    # halfday_dict = {0:'bl1_light',1:'bl1_dark',
    #                 2:'bl2_light',3:'bl2_dark'
    #                 4:'sd_light',5:'sd_dark'
    #                 6:'r1_light',7:'r1_dark'}
    cata_count_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    cata_duration_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    wake_pre_cata_duration_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    #
    # for mouse in mice :
    #     # print(mouse)
    #     for h, i in enumerate(np.arange(0, number_of_epochs, window)):
    #         # print('hour = ',h)
    #         subs = cata[mouse][i : i+window]
    #         if subs.sum() == 0 :
    #             # print( 'yep   ', h)
    #             cata_count_by_halfday.at[h, mouse]=0
    #         else :
    #             counter = 0
    #             cata_number = []
    #             for i in subs:
    #                 if i ==0:
    #                     cata_number.append(counter)
    #                     counter = 0
    #                 if i ==1:
    #                     counter +=1
    #             # cata_number.append(counter) #### !!!! Add last overlapping event would count twice the same event : end of hour h and begin of hour h+1
    #             cata_number=np.array(cata_number)
    #             mask = cata_number!=0
    #             cata_number = cata_number[mask]
    #             cata_duration = np.mean(cata_number*4/60)
    #             cata_duration_by_hour.at[h, mouse]= cata_duration
    #             cata_count_by_halfday.at[h, mouse]= cata_number.size

    for mouse in mice :
        # print(mouse)
        for h, hour in enumerate(np.arange(0, number_of_epochs, window)):
            # print('hour = ',h)
            subs = cata[mouse][hour : hour+window].values
            sub_wake = wake[mouse][hour : hour+window].values
            positions =[]
            wake_pre_cata_duration = []

            if subs.sum() == 0 :
                # print( 'yep   ', h)
                cata_count_by_hour.at[h, mouse]=0
                cata_duration_by_hour.at[h, mouse]=0
            else :
                counter = 0
                cata_event_size = []
                for pos,i in enumerate(subs):
                    if i ==0 and counter !=0:
                        cata_event_size.append(counter)
                        counter = 0
                        ini = positions[0]
                        if ini !=0 :
                            # print(ini)
                            if sub_wake[ini-1]!= 1 and subs[ini-1]!= 1:
                                print('##### ENCODING ERROR ##### cataplexy during sleep')
                                sub_rem = rem[mouse][hour : hour+window].values
                                if  sub_rem[ini-1] ==1:
                                    print('REM')
                                sub_nrem = nrem[mouse][hour : hour+window].values
                                if  sub_nrem[ini-1] ==1:
                                    print('NREM')
                            else :
                                inverted_wake_for_counting = sub_wake[:ini]
                                inverted_wake_for_counting = inverted_wake_for_counting[::-1]
                                wake_count = 0
                                for w in inverted_wake_for_counting:
                                    if w == 0:
                                        break
                                    else :
                                        wake_count +=1
                                wake_pre_cata_duration.append(wake_count)
                    if i ==1:
                        counter +=1
                        positions.append(pos)
                # cata_number.append(counter) #### !!!! Add last overlapping event would count twice the same event : end of hour h and begin of hour h+1
                cata_event_size=np.array(cata_event_size)
                mask = cata_event_size!=0
                cata_event_size = cata_event_size[mask]
                wake_pre_cata_duration = np.mean(wake_pre_cata_duration)*4/60
                cata_duration = np.mean(cata_event_size*4/60)

                cata_count_by_hour.at[h, mouse]= cata_event_size.size
                cata_duration_by_hour.at[h, mouse]= cata_duration
                wake_pre_cata_duration_by_hour.at[h, mouse]= wake_pre_cata_duration

    dirname = excel_dir + '/Ox1r_Ox2r_dd/cataplexy_count/'
    filename =dirname+'by_halfday_cataplexy_count.xlsx'
    cata_count_by_hour.to_excel(filename)

    dirname = excel_dir + '/Ox1r_Ox2r_dd/cataplexy_duration/'
    filename =dirname+'by_halfday_cataplexy_duration.xlsx'
    cata_duration_by_hour.to_excel(filename)

    dirname = excel_dir + '/Ox1r_Ox2r_dd/wake_pre_cataplexy_duration/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename =dirname+'by_halday_cataplexy_duration.xlsx'
    wake_pre_cata_duration_by_hour.to_excel(filename)

def plot_tdw_before_cata(spectrum_method = 'welch'):
    # data_dir = 'C:/Users/maxime.juventin/Desktop/scripts_ML/data/'
    path = precompute_dir + '/Ox1r_Ox2r_dd/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)

    # print(ds)
    number_of_epochs = ds['wake_all_mice'].shape[0]
    wake = ds['wake_all_mice'].to_pandas()
    rem = ds['rem_all_mice'].to_pandas()
    tdw = ds['tdw_all_mice'].to_pandas()
    nrem = ds['nrem_all_mice'].to_pandas()
    times = ds['epochs'].values*4/3600

    # epoch_duration = 4
    # duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    # print(ds['cata_all_mice'].to_pandas())
    cata = ds['cata_all_mice'].to_pandas()
    # print(cata.shape)

    score = pd.DataFrame( np.ones(cata.shape).astype('str'), columns = ds['mice'].values,index = ds['epochs'].values)
    for mouse in ds['mice'].values:
        # line = score[r]
        # print(line)
        # line[wake[r] ==1] = np.array(['w']*np.sum(wake[r] ==1))
        # print(mouse)
        score[mouse][wake[mouse] ==1] = np.array(['w']*np.sum(wake[mouse] ==1))
        score[mouse][cata[mouse] ==1] = np.array(['a']*np.sum(cata[mouse] ==1))
        score[mouse][nrem[mouse] ==1] = np.array(['n']*np.sum(nrem[mouse] ==1))
        score[mouse][rem[mouse] ==1] = np.array(['r']*np.sum(rem[mouse] ==1))
        score[mouse][tdw[mouse] ==1] = np.array(['t']*np.sum(tdw[mouse] ==1))
        # print(score)
        # exit()
    # time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    # window = int(time * 3600 / epoch_duration)
    # cata_by_time = np.zeros(int(duration_in_hours/time))

    mice = cata.columns.to_list()
    dark_periods = {'dark1' :(12,24), 'dark2':(36, 48), 'dark3':(50, 72), 'dark4':(84, 96)}
    # time = int(12) # window to look at in HOUR. CAUTION must be a multiple of 100
    # window = int(time * 3600 / epoch_duration)
    # cata_by_time = np.zeros(int(duration_in_hours/time))
    # # halfday_dict = {0:'bl1_light',1:'bl1_dark',
    # #                 2:'bl2_light',3:'bl2_dark'
    # #                 4:'sd_light',5:'sd_dark'
    # #                 6:'r1_light',7:'r1_dark'}
    # cata_count_by_halfday = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    # cata_duration_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    # wake_pre_cata_duration_by_hour = pd.DataFrame( np.zeros((1,cata.shape[1])), columns = cata.columns.to_list())
    #
    # for mouse in mice :
    #     # print(mouse)
    #     for h, i in enumerate(np.arange(0, number_of_epochs, window)):
    #         # print('hour = ',h)
    #         subs = cata[mouse][i : i+window]
    #         if subs.sum() == 0 :
    #             # print( 'yep   ', h)
    #             cata_count_by_halfday.at[h, mouse]=0
    #         else :
    #             counter = 0
    #             cata_number = []
    #             for i in subs:
    #                 if i ==0:
    #                     cata_number.append(counter)
    #                     counter = 0
    #                 if i ==1:
    #                     counter +=1
    #             # cata_number.append(counter) #### !!!! Add last overlapping event would count twice the same event : end of hour h and begin of hour h+1
    #             cata_number=np.array(cata_number)
    #             mask = cata_number!=0
    #             cata_number = cata_number[mask]
    #             cata_duration = np.mean(cata_number*4/60)
    #             cata_duration_by_hour.at[h, mouse]= cata_duration
    #             cata_count_by_halfday.at[h, mouse]= cata_number.size
    mean_tdw_percent = pd.DataFrame(index = list(dark_periods.keys()), columns = ds['mice'].values)
    for period in dark_periods:
        fig, ax = plt.subplots()

        for mouse in mice :

            ds_spec = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
            # print(ds_spec)
            spectrum = ds_spec['{}_spectrum'.format(spectrum_method)].values
            freqs = ds_spec.coords['freqs_{}'.format(spectrum_method)].values

        # print(mouse)
        # for h, hour in enumerate(np.arange(0, number_of_epochs, window)):
            # print('hour = ',h)
            mask = (times>dark_periods[period][0]) &  (times<dark_periods[period][1])
            subs = cata[mouse][mask].values
            sub_spec = spectrum[mask]
            sub_score = score[mouse][mask].values
            # sub_wake = wake[mouse][mask].values
            positions =[]
            wake_pre_cata_duration = 15#### 1min

            if subs.sum() == 0 :
                # print( 'yep   ', h)
                # cata_count_by_hour.at[h, mouse]=0
                # cata_duration_by_hour.at[h, mouse]=0
                mean_tdw_percent.at[period, mouse]=0
            else :
                counter = 0
                tdw_percent_before = []
                spectrum_before = []
                for pos,i in enumerate(subs):
                    if i ==0 and counter !=0:
                        counter = 0
                        ini = positions[0]
                        if ini !=0 :
                            before = sub_score[ini-10: ini]
                            possible_states = np.unique(before)
                            positions = []
                            if 'r' in possible_states:
                                print('##### ENCODING ERROR ##### cataplexy during sleep')
                                print('REM')
                            elif 'n' in possible_states:
                                print('##### ENCODING ERROR ##### cataplexy during sleep')
                                print('NREM')
                            else :
                                spectrum_before.append(sub_spec[ini-10: ini,:])
                                tdw_percent_before.append(100*np.sum(before=='t')/before.size)

                    if i ==1:
                        counter +=1
                        positions.append(pos)
                mean_tdw_percent.at[period, mouse]=np.mean(tdw_percent_before)
                if len(spectrum_before) !=0 :
                    spectrum_before = np.concatenate(spectrum_before)
                    ax.plot(freqs, np.mean(spectrum_before, axis = 0))

                # # cata_number.append(counter) #### !!!! Add last overlapping event would count twice the same event : end of hour h and begin of hour h+1
                # cata_event_size=np.array(cata_event_size)
                # mask = cata_event_size!=0
                # cata_event_size = cata_event_size[mask]
                # wake_pre_cata_duration = np.mean(wake_pre_cata_duration)*4/60
                # cata_duration = np.mean(cata_event_size*4/60)
                #
                # cata_count_by_hour.at[h, mouse]= cata_event_size.size
                # cata_duration_by_hour.at[h, mouse]= cata_duration
                # wake_pre_cata_duration_by_hour.at[h, mouse]= wake_pre_cata_duration

        # plt.show()
    dirname = excel_dir + '/Ox1r_Ox2r_dd/tdw_before_cata/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename =dirname+'tdw_percent.xlsx'
    mean_tdw_percent.to_excel(filename)



# def plot_compare_sleep_state_accross_time(control ='Control', test ='DCR_HCRT'):
#     # data_dir = 'C:/Users/maxime.juventin/Desktop/scripts_ML/data/'
#     path_control = precompute_dir +  '/' +control+'/sleep_by_epoch.nc'
#     path_cre = precompute_dir + '/'+ test+'/sleep_by_epoch.nc'
#     ds_control = xr.open_dataset(path_control)
#     ds_cre = xr.open_dataset(path_cre)
#
#     number_of_epochs = ds_cre['wake_all_mice'].shape[0]
#     epoch_duration = 4
#     duration_in_hours = int(number_of_epochs*epoch_duration/3600)
#     wake_control = ds_control['wake_all_mice'].to_pandas()
#     nrem_control = ds_control['nrem_all_mice'].to_pandas()
#     rem_control = ds_control['rem_all_mice'].to_pandas()
#     cata_control = ds_control['cata_all_mice'].to_pandas()
#
#     wake_cre = ds_cre['wake_all_mice'].to_pandas()
#     nrem_cre = ds_cre['nrem_all_mice'].to_pandas()
#     rem_cre = ds_cre['rem_all_mice'].to_pandas()
#     cata_cre = ds_cre['cata_all_mice'].to_pandas()
#
#     time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
#     window = int(time * 3600 / epoch_duration)
#     wake_by_time_control = []
#     rem_by_time_control = []
#     nrem_by_time_control = []
#     cata_by_time_control = []
#
#     wake_by_time_cre = []
#     rem_by_time_cre = []
#     nrem_by_time_cre = []
#     cata_by_time_cre = []
#
#     for h, i in enumerate(np.arange(0, number_of_epochs, window)):
#
#         wake_by_time_control.append(wake_control[i : i+window].sum())
#         rem_by_time_control.append(rem_control[i : i+window].sum())
#         nrem_by_time_control.append(nrem_control[i : i+window].sum())
#         cata_by_time_control.append(cata_control[i : i+window].sum())
#
#         wake_by_time_cre.append(wake_cre[i : i+window].sum())
#         rem_by_time_cre.append(rem_cre[i : i+window].sum())
#         nrem_by_time_cre.append(nrem_cre[i : i+window].sum())
#         cata_by_time_cre.append(cata_cre[i : i+window].sum())
#     wake_by_time_control = pd.concat(wake_by_time_control, axis = 1)
#     cata_by_time_control = pd.concat(cata_by_time_control, axis = 1)
#     rem_by_time_control = pd.concat(rem_by_time_control, axis = 1)
#     nrem_by_time_control = pd.concat(nrem_by_time_control, axis = 1)
#     wake_by_time_cre = pd.concat(wake_by_time_cre, axis = 1)
#     rem_by_time_cre = pd.concat(rem_by_time_cre, axis = 1)
#     nrem_by_time_cre = pd.concat(nrem_by_time_cre, axis = 1)
#     cata_by_time_cre = pd.concat(cata_by_time_cre, axis = 1)
#
#     # print(wake_by_time.shape)
#     fig, ax = plt.subplots(nrows=3)
#     times = np.arange(0, duration_in_hours)
#     plot  = np.arange(3)
#     controls = [wake_by_time_control, rem_by_time_control, nrem_by_time_control, cata_by_time_control]
#     cres = [wake_by_time_cre, rem_by_time_cre, nrem_by_time_cre, cata_by_time_cre]
#     # colors = [ 'black', 'red', 'blue']
#     # colors = [ 'black', 'green', 'black', 'green', 'black', 'green']
#     labels = ['control', 'control' ,'control', 'control', 'DCR', 'DCR', 'DCR' 'DCR']
#
#     for i, data_control, data_cre, label in zip(plot, controls, cres,labels) :
#         m = data_control.mean(axis=0)*epoch_duration/60
#         s =data_control.std(axis=0)*epoch_duration/60
#         ax[i].plot(times, m, color = 'black', label = label)
#         ax[i].fill_between(times, m-s, m+s , color = 'black', alpha = .2,)
#         m = data_cre.mean(axis=0)*epoch_duration/60
#         s =data_cre.std(axis=0)*epoch_duration/60
#         ax[i].plot(times, m, color = 'green', label = label)
#         ax[i].fill_between(times, m-s, m+s , color = 'green', alpha = .2)
#
#
#     nights = { 'dark0' : [0, 4],
#                  'dark1' : [16, 28],
#                  'dark2' : [40, 52],
#                  'dark3':[64, 76],
#                  'dark4':[88, 100]}
#
#     for night in nights :
#         h1, h2 = nights[night][0], nights[night][1]
#         ax[0].axvspan(h1, h2 , color = 'black', alpha = .3)
#         ax[1].axvspan(h1, h2 , color = 'black', alpha = .3)
#         ax[2].axvspan(h1, h2 , color = 'black', alpha = .3)
#         ax[3].axvspan(h1, h2 , color = 'black', alpha = .3)
#     height0_1, height0_2 =ax[0].get_ylim()[1]*.95, ax[0].get_ylim()[1]*1.05
#     height1_1, height1_2 = ax[1].get_ylim()[1]*.95, ax[1].get_ylim()[1]*1.05
#     height2_1, height2_2 = ax[2].get_ylim()[1]*.95, ax[2].get_ylim()[1]*1.05
#     height3_1, height3_2 = ax[3].get_ylim()[1]*.95, ax[3].get_ylim()[1]*1.05
#     for x1, x2 in zip([0,28,76], [4,52,100]):
#         lim = 110
#         x1 += 5
#         x2 +=5
#         ax[0].axhspan(height0_1, height0_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
#         ax[1].axhspan(height1_1, height1_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
#         ax[2].axhspan(height2_1, height2_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
#         ax[2].axhspan(height3_1, height3_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
#     x1 = 52 + 5
#     x2 = 58 +5
#     ax[0].axhspan(height0_1, height0_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
#     ax[1].axhspan(height1_1, height1_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
#     ax[2].axhspan(height2_1, height2_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
#     ax[3].axhspan(height3_1, height3_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
#         # ax[0].axhspan(i,j , color = 'black', alpha = .3)
#         # ax[1].axhspan(i,j , color = 'black', alpha = .3)
#         # ax[2].axhspan(i,j , color = 'black', alpha = .3)
#     ax[0].set_title('Sleep state amount DCR vs control for , sum per ' + str(time) + ' hours')
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     ax[3].legend()
#     ax[0].set_ylabel('wake')
#     ax[1].set_ylabel('rem')
#     ax[2].set_ylabel('nrem')
#     ax[3].set_ylabel('cata')
#
#     plt.show()

def sleep_bouts(group):
    # B0 begins at 4h, B1 etc begins at 8
    path = precompute_dir + group + '/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)

    all_mice_w_epoch = ds['wake_all_mice'].to_pandas()
    all_mice_t_epoch = ds['tdw_all_mice'].to_pandas()
    all_mice_n_epoch = ds['nrem_all_mice'].to_pandas()
    all_mice_cata_epoch = ds['cata_all_mice'].to_pandas()
    all_mice_r_epoch = ds['rem_all_mice'].to_pandas()    #.unstack().T*1
    data_by_state = {'wake':all_mice_w_epoch,
                    'nrem' : all_mice_n_epoch,
                    'tdw' : all_mice_t_epoch,
                    'rem' :all_mice_r_epoch,
                    'cata':all_mice_cata_epoch}
    number_of_epochs = ds['wake_all_mice'].shape[0]
    # print(number_of_epochs)


    # x = ds['epochs'].values
    # fig, ax = plt.subplots()
    #
    # ax.scatter(x[all_mice_t_epoch.values[:,1]==1], 5*np.ones(np.sum(all_mice_t_epoch.values[:,1])))
    # ax.scatter(x[all_mice_w_epoch.values[:,1]==1], 4*np.ones(np.sum(all_mice_w_epoch.values[:,1])))
    # ax.scatter(x[all_mice_n_epoch.values[:,1]==1], 3*np.ones(np.sum(all_mice_n_epoch.values[:,1])))
    # ax.scatter(x[all_mice_r_epoch.values[:,1]==1], 2*np.ones(np.sum(all_mice_r_epoch.values[:,1])))
    # ax.scatter(x[all_mice_cata_epoch.values[:,1]==1], 1*np.ones(np.sum(all_mice_cata_epoch.values[:,1])))
    # plt.show()
    # exit()


    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    # protocol = ['bl1', 'bl2', 'sd', 'sr']
    # hours = np.array(([4, 28], [28, 52], [52,76], [76, 100]) )
    # days_info = { 'bl1' : [4, 28], 'bl2' : [28, 52], 'sd' : [52,76], 'sr':[76, 100]}
    # day_cycles = { 'dark1' : [4, 4+8],
    #              'light1' : [4+8, 4+20],
    #              'dark2' : [4+20, 28+8],
    #              'light2' : [28+8, 28+20],
    #              'dark3' : [28+20, 52+8],
    #              'sd' : [52+8, 52+14],
    #              'light3' : [52+14,52+20],
    #              'dark4':[52+20, 76+8],
    #              'light4':[76+8, 76+20],
    #              'dark5':[76+20, 100]}
    day_cycles = {
                 'light1' : [0, 12],
                 'dark1' : [12, 24],
                 'light2' : [24, 36],
                 'dark2' : [36, 48],
                 'sd' : [48, 54],
                 'light3' : [54,60],
                 'dark3':[60, 72],
                 'light4':[72,84 ],
                 'dark4':[84, 96]}
    # time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    # window = int(time * 3600 / epoch_duration)
    # hours_in_epochs = hours*3600/epoch_duration
    # print(hours_in_epochs)
    mice = ds.coords['mice'].values

    for s, state in enumerate(data_by_state):
        data = data_by_state[state]
        for c, cycle in enumerate(day_cycles):
            cycle_hours = day_cycles[cycle]
            # print(cycle_hours)
            i1, i2 = int(cycle_hours[0]), int(cycle_hours[1])
            i1 = int(i1 *3600/epoch_duration)
            i2 = int(i2 *3600/epoch_duration)
            selected_data = data[i1: i2]
            # print(diff.where(diff == -1))
            all_mice_count = {}
            all_mice_percent_frag = {}
            all_mice_mean_bout_duration = {}
            all_mice_bout_amount = {}
            all_mice_state_time = {}
            # print('je suis la')
            for mouse in mice :
                one_and_zeros_one_mouse = selected_data[mouse].values
                counter = 0
                bouts = []
                state_time = np.sum(one_and_zeros_one_mouse)*4/60  #min
                all_mice_state_time[mouse] = state_time

                for i in one_and_zeros_one_mouse:
                    if i ==0:
                        bouts.append(counter)
                        counter = 0
                    if i ==1:
                        counter +=1

                #### True duration last bout unknown
                bouts.append(counter)
                bouts=np.array(bouts)
                mask = bouts!=0
                bouts = bouts[mask]
                all_mice_bout_amount[mouse] = [len(bouts), len(bouts)/12]

                bouts = bouts*4
                mean_bouts = np.mean(bouts)/60 #min
                all_mice_mean_bout_duration[mouse] = mean_bouts

                if bouts.size >0 :
                    if np.max(bouts) > 2**12:
                        bins_sleep = np.append(2**np.arange(2,12), np.max(bouts))
                    else :
                        bins_sleep = 2**np.arange(2,13)
                    bins_sleep= np.append(np.array([0]), bins_sleep)

                    #bins_sleep += [max(bouts)]
                    # count, bins= np.histogram(bouts, bins = bins_sleep)
                    state_percentage_frag = []
                    count = []
                    for b in np.arange(1,bins_sleep.size):
                        # print(bins_sleep[b])
                        # print(bins_sleep[b+1])
                        # print()
                        mask = (bouts>bins_sleep[b-1]) & (bouts<=bins_sleep[b])
                        # print(np.unique(bouts[mask]))
                        # print(bins_sleep[b])
                        state_percentage_frag.append(np.sum(bouts[mask])/(60*state_time))
                        count.append(np.sum(mask))

                    state_percentage_frag = np.array(state_percentage_frag)
                    count = np.array(count)
                    # print(count.size)
                    # print(bins_sleep.size)
                    # exit()
                else :
                    bins_sleep = 2**np.arange(2,13)
                    bins_sleep= np.append(np.array([0]), bins_sleep)
                    count = np.zeros(np.arange(1,13).size-1)
                    state_percentage_frag = np.zeros(np.arange(1,13).size-1)

                all_mice_count[mouse] = count
                all_mice_percent_frag[mouse] = state_percentage_frag
                # fig, ax = plt.subplots()
                # ax.plot(count)
                # axb=ax.twinx()
                # axb.plot(state_percentage_frag, color ='k')
                # plt.show()
                # exit()
            bins_sleep = 2**np.arange(2,13)
            df_percent_frag = pd.DataFrame.from_dict(all_mice_percent_frag, orient = 'index', columns = bins_sleep )
            output_path = excel_dir + '/'+ group + '/sleep_percent_fragmentation/' + state + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df_percent_frag.to_excel(output_path + state + '_' + cycle + '.xlsx')


            df = pd.DataFrame.from_dict(all_mice_count, orient = 'index', columns = bins_sleep )
            output_path = excel_dir + '/'+ group + '/sleep_fragmentation/' + state + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df.to_excel(output_path + state + '_' + cycle + '.xlsx')

            df_duration = pd.DataFrame.from_dict(all_mice_mean_bout_duration, orient = 'index', columns = ['mean_duration'] )
            output_path = excel_dir + '/'+ group + '/state_duration/' + state + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df_duration.to_excel(output_path + state + '_' + cycle + '.xlsx')

            df_time = pd.DataFrame.from_dict(all_mice_state_time, orient = 'index', columns = ['time'] )
            output_path = excel_dir + '/'+ group + '/state_time/' + state + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df_time.to_excel(output_path + state + '_' + cycle + '.xlsx')

            df_period_amount = pd.DataFrame.from_dict(all_mice_bout_amount, orient = 'index', columns = ['period', 'freq'] )
            output_path = excel_dir + '/'+ group + '/period_amount/' + state + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df_period_amount.to_excel(output_path + state + '_' + cycle + '.xlsx')


##### TODO refaire a zero avec new indication MALO et maintenant wake = w + t
# def REM_sleep_latency_one_mouse(mouse):
#     min_wake_duration = 6
#     ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
#     times = ds['times_somno'].values/3600
#     score = ds['score'].values
#     score_behavior = score.copy()
#     ctrl = get_mice('Control')
#     dcr = get_mice('Ox1r_Ox2r_d/d')
#     group_mice = {'Control' : ctrl, 'Ox1r_Ox2r_d/d':dcr}
#     if mouse in group_mice['Control'] :
#         group = 'Control'
#     elif mouse in group_mice['Ox1r_Ox2r_d/d'] :
#         group = 'Ox1r_Ox2r_d/d'
#
#
#     halfday_times= {
#                  'light1' : [0, 12],
#                  'dark1' : [12, 24],
#                  'light2' : [24, 36],
#                  'dark2' : [36, 48],
#                  'sd' : [48, 54],
#                  'light3' : [54,60],
#                  'dark3':[60, 72],
#                  'light4':[72,84 ],
#                  'dark4':[84, 96]}
#
#     for f, n in zip(['1', '2', '3'], ['w', 'n', 'r']) :
#         score_behavior = np.where(score_behavior == f, n, score_behavior)
#
#     for halfday in halfday_times:
#         dirname = excel_dir + '/{}/REM_latency/'.format(group)
#         if not os.path.exists(dirname):
#             os.makedirs(dirname)
#         filename =dirname+'REM_latency_{}_{}.xlsx'.format(group, halfday)
#         if not os.path.exists(filename):
#             df = pd.DataFrame(index = group_mice[group], columns = np.arange(5))
#         else :
#             df = pd.read_excel(filename, index_col = 0)
#
#
#         t1 = halfday_times[halfday][0]
#         t2 = halfday_times[halfday][1]
#         time_mask = (times>t1) & (times<t2)
#
#         masked_score = score_behavior[time_mask]
#         rem = score_behavior[time_mask] == 'r'
#         nrem = score_behavior[time_mask] == 'n'
#         wake = score_behavior[time_mask] == 'w'
#         cata = score_behavior[time_mask] == 'a'
#
#         # fig, ax = plt.subplots()
#         latencies = []
#         ini_rem = np.diff(rem*1)
#         pos_ini_rem = np.where(ini_rem == 1)[0]+1
#         for pos in pos_ini_rem:
#             wake_duration = 0
#             for latency, ind in enumerate(np.arange(pos-1)[::-1]):
#                 s = masked_score[ind]
#                 if s == 'w':
#                     wake_duration += 1
#                 if s == 'w' and wake_duration >=min_wake_duration:
#                     if latency == 5:
#                         pos_ini_rem = pos_ini_rem[pos_ini_rem!=pos]
#                         break
#                     else :
#                         latency -= 5
#                         latencies.append(latency)
#                         break
#                 elif s == 'n' and wake_duration < min_wake_duration:
#                     wake_duration = 0
#                 elif s =='r' and wake_duration == latency:
#                     pos_ini_rem = pos_ini_rem[pos_ini_rem!=pos]
#                     break
#                 elif s == 'r' and wake_duration != latency:
#                     if masked_score[ind+1] == 'w':
#                         latency -= wake_duration
#                         latencies.append(latency)
#                     else :
#                         latencies.append(latency)
#                     break
#                     # break
#
#             # ax.plot(np.arange(pos-latency, pos), np.ones(latency), color = 'black')
#         # print(df.columns.to_list())
#         # print(len(latencies))
#         # print(latencies)
#         if len(df.columns.to_list())<len(latencies):
#             df = df.reindex(columns = np.arange(len(latencies)))
#         df.loc[mouse, np.arange(len(latencies))] = np.array(latencies)
#         df.to_excel(filename)

def inter_REM_interval(group):
    # B0 begins at 4h, B1 etc begins at 8
    path = precompute_dir + group + '/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)

    all_mice_r_epoch = ds['rem_all_mice'].to_pandas()    #.unstack().T*1
    number_of_epochs = ds['rem_all_mice'].shape[0]

    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    day_cycles = {
                 'light1' : [0, 12],
                 'dark1' : [12, 24],
                 'light2' : [24, 36],
                 'dark2' : [36, 48],
                 'sd' : [48, 54],
                 'light3' : [54,60],
                 'dark3':[60, 72],
                 'light4':[72,84 ],
                 'dark4':[84, 96]}

    mice = ds.coords['mice'].values

    data = all_mice_r_epoch
    for c, cycle in enumerate(day_cycles):
        cycle_hours = day_cycles[cycle]
        # print(cycle_hours)
        i1, i2 = int(cycle_hours[0]), int(cycle_hours[1])
        i1 = int(i1 *3600/epoch_duration)
        i2 = int(i2 *3600/epoch_duration)
        selected_data = data[i1: i2]
        # print(diff.where(diff == -1))
        all_mice_IRI = {}

        # print('je suis la')
        for mouse in mice :
            one_and_zeros_one_mouse = selected_data[mouse].values
            # print(np.sum(one_and_zeros_one_mouse))
            diff = np.diff(one_and_zeros_one_mouse)
            ini = np.where(diff ==1)[0]+1
            if one_and_zeros_one_mouse[0]==1:
                ini = np.append(np.array([0]), ini)
            end = np.where(diff ==-1)[0]
            if one_and_zeros_one_mouse[-1]==1:
                end = np.append(end, np.array([one_and_zeros_one_mouse.size]))
            IRI = np.mean(ini[1:]-end[:-1])
            IRI *= (4/60)
            all_mice_IRI[mouse]=IRI
            # fig, ax = plt.subplots()
            # ax.plot(diff)
            # ax.plot(one_and_zeros_one_mouse)
            # ax.scatter(ini, np.ones(ini.size), color ='green')
            # ax.scatter(end, np.ones(end.size), color ='red')
            # for i,iri in enumerate(IRI):
            #     ax.plot([end[:-1][i], end[:-1][i]+iri],[1,1])
            # plt.show()
            # exit()

        df_IRI = pd.DataFrame.from_dict(all_mice_IRI, orient = 'index', columns = ['IRI'] )
        output_path = excel_dir + '/'+ group + '/IRI/' + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df_IRI.to_excel(output_path  + cycle + '.xlsx')
#
# def compute_all_REM_latency():
#     mice = get_mice('Control') + get_mice('Ox1r_Ox2r_d/d')
#     for mouse in mice :
#         print(mouse)
#         REM_sleep_latency_one_mouse(mouse)

def REM_sleep_latency_one_mouse(mouse):
    min_wake_duration = 6
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds['times_somno'].values/3600
    score = ds['score'].values
    score_behavior = score.copy()
    df_excel = pd.read_excel(work_dir + 'datetime_reference_OXR.xlsx', index_col = 0)
    group = df_excel.at[mouse, 'group']
    mice = get_mice(group)


    halfday_times= {
                 'light1' : [0, 12],
                 'dark1' : [12, 24],
                 'light2' : [24, 36],
                 'dark2' : [36, 48],
                 'sd' : [48, 54],
                 'light3' : [54,60],
                 'dark3':[60, 72],
                 'light4':[72,84 ],
                 'dark4':[84, 96]}

    for f, n in zip(['1', '2', '3'], ['w', 'n', 'r']) :
        score_behavior = np.where(score_behavior == f, n, score_behavior)

    for halfday in halfday_times:
        dirname = excel_dir + '/{}/REM_latency/'.format(group)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename =dirname+'REM_latency_{}_{}.xlsx'.format(group, halfday)
        if not os.path.exists(filename):
            df = pd.DataFrame(index = mice, columns = np.arange(5))
        else :
            df = pd.read_excel(filename, index_col = 0)


        t1 = halfday_times[halfday][0]
        t2 = halfday_times[halfday][1]
        time_mask = (times>t1) & (times<t2)

        masked_score = score_behavior[time_mask]
        rem = score_behavior[time_mask] == 'r'
        nrem = score_behavior[time_mask] == 'n'
        wake = score_behavior[time_mask] == 'w'

        # fig, ax = plt.subplots()
        latencies = []
        ini_rem = np.diff(rem*1)
        pos_ini_rem = np.where(ini_rem == 1)[0]+1
        for pos in pos_ini_rem:
            wake_duration = 0
            for latency, ind in enumerate(np.arange(pos-1)[::-1]):
                s = masked_score[ind]
                if s == 'w':
                    wake_duration += 1
                if s == 'w' and wake_duration >=min_wake_duration:
                    if latency == 5:
                        pos_ini_rem = pos_ini_rem[pos_ini_rem!=pos]
                        break
                    else :
                        latency -= 5
                        latencies.append(latency)
                        break
                elif s == 'n' and wake_duration < min_wake_duration:
                    wake_duration = 0
                elif s =='r' and wake_duration == latency:
                    pos_ini_rem = pos_ini_rem[pos_ini_rem!=pos]
                    break
                elif s == 'r' and wake_duration != latency:
                    if masked_score[ind+1] == 'w':
                        latency -= wake_duration
                        latencies.append(latency)
                    else :
                        latencies.append(latency)
                    break
                    # break

            # ax.plot(np.arange(pos-latency, pos), np.ones(latency), color = 'black')
        # print(df.columns.to_list())
        # print(len(latencies))
        # print(latencies)
        if len(df.columns.to_list())<len(latencies):
            df = df.reindex(columns = np.arange(len(latencies)))
        df.loc[mouse, np.arange(len(latencies))] = np.array(latencies)*4/60
        df.to_excel(filename)


def REM_sleep_latency_by_hour(group):
    min_wake_duration = 6
    epoch_duration = 4

    mice = get_mice(group)
    df = pd.DataFrame(index = mice, columns = np.arange(96))
    for mouse in mice :
        ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
        times = ds['times_somno'].values/3600
        score = ds['score'].values

        score_behavior = score.copy()
        # miRNA, genotype = get_mouse_info(mouse)

        for f, n in zip(['1', '2', '3'], ['w', 'n', 'r']) :
            score_behavior = np.where(score_behavior == f, n, score_behavior)
        hours = np.arange(0, int(round(times[-1])))

        window = int(3600 / epoch_duration)

        print(hours)
        # exit()
        c = 1
        for hour in hours :
            i1, i2 = int(hour*window), int((hour+1)*window)
            print(c)
            c+=1
            print(hour, hour+1)

            t1 = hour
            t2 = hour+1
            time_mask = (times>t1) & (times<t2)
            print(np.sum(time_mask))
            print()
            print()
            masked_score = score_behavior[time_mask]
            rem = score_behavior[time_mask] == 'r'
            nrem = score_behavior[time_mask] == 'n'
            wake = score_behavior[time_mask] == 'w'

            # fig, ax = plt.subplots()
            latencies = []
            ini_rem = np.diff(rem*1)
            pos_ini_rem = np.where(ini_rem == 1)[0]+1
            for pos in pos_ini_rem:
                wake_duration = 0
                for latency, ind in enumerate(np.arange(pos-1)[::-1]):
                    s = masked_score[ind]
                    if s == 'w':
                        wake_duration += 1
                    if s == 'w' and wake_duration >=min_wake_duration:
                        if latency == 5:
                            pos_ini_rem = pos_ini_rem[pos_ini_rem!=pos]
                            break
                        else :
                            latency -= 5
                            latencies.append(latency)
                            break
                    elif s == 'n' and wake_duration < min_wake_duration:
                        wake_duration = 0
                    elif s =='r' and wake_duration == latency:
                        pos_ini_rem = pos_ini_rem[pos_ini_rem!=pos]
                        break
                    elif s == 'r' and wake_duration != latency:
                        if masked_score[ind+1] == 'w':
                            latency -= wake_duration
                            latencies.append(latency)
                        else :
                            latencies.append(latency)
                        break
                        # break
            latencies = 4*np.array(latencies)
            if len(latencies)>0:
                latency = np.mean(latencies)*4/60
            else :
                latency = np.nan
            df.loc[mouse, hour] = latency

        # exit()
    dirname = excel_dir + '/{}/REM_latency_by_hour/'.format(group)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    df.to_excel(dirname + '/REM_latency_by_hour.xlsx')


def compute_all_REM_latency():
            for group in groups:
                REM_sleep_latency_by_hour(group)
                for mouse in get_mice(group):
                    REM_sleep_latency_one_mouse(mouse)


if __name__ == '__main__' :
    mouse = 'B4906'


    # group = "Ox1r_Ox2r_d/d"
    #group = "Control"
    # plot_tdw_before_cata()
    # precompute_sleep_state_by_epoch(group)
    # sleep_state_statistics(group)
    # sleep_bouts(group)
    # REM_sleep_latency_by_hour(group)
    # inter_REM_interval(group)
    # REM_sleep_latency_one_mouse(mouse)
    # exit()
    # plot_cata_number_accross_time()
    # plot_tdw_before_cata()
    compute_all()
    compute_all_REM_latency()
