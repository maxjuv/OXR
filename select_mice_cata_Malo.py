from configuration import *


def get_mice(group):
    df_excel = pd.read_excel(work_dir + 'datetime_reference_OXR.xlsx', index_col = 0)
    mylist = df_excel[df_excel.group == group].index.to_list()
    # mice = [i[4:] for i in mylist]
    mice = [int(i) for i in mylist]
    return mice

def get_all_mice():
    mice = []
    for group in groups :
        mice+=get_mice(group)
    return mice

def get_mouse_info(mouse):
    df_excel = pd.read_excel(work_dir + 'datetime_reference_OXR.xlsx', index_col = 0)
    mouse = df_excel.at[mouse,'group']
    return mouse



if __name__ == '__main__':
    for group in groups :
        print(get_mice(group))
    print(get_all_mice())


    # print(get_mice_for_spectrum('Control'))
    # print(get_mice_for_spectrum('Ataxine'))
