def StratifiedGroupShuffleSplit(df_main):

    df_main = df_main.reindex(np.random.permutation(df_main.index)) # shuffle dataset

    # create empty train, val and test datasets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    hparam_mse_wgt = 0.1 # must be between 0 and 1
    assert(0 <= hparam_mse_wgt <= 1)
    train_proportion = 0.6 # must be between 0 and 1
    assert(0 <= train_proportion <= 1)
    val_test_proportion = (1-train_proportion)/2

    subject_grouped_df_main = df_main.groupby(['subject_id'], sort=False, as_index=False)
    category_grouped_df_main = df_main.groupby('category').count()[['subject_id']]/len(df_main)*100

    def calc_mse_loss(df):
        grouped_df = df.groupby('category').count()[['subject_id']]/len(df)*100
        df_temp = category_grouped_df_main.join(grouped_df, on = 'category', how = 'left', lsuffix = '_main')
        df_temp.fillna(0, inplace=True)
        df_temp['diff'] = (df_temp['subject_id_main'] - df_temp['subject_id'])**2
        mse_loss = np.mean(df_temp['diff'])
        return mse_loss

    i = 0
    for _, group in subject_grouped_df_main:

        if (i < 3):
            if (i == 0):
                df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            elif (i == 1):
                df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            else:
                df_test = df_test.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue

        mse_loss_diff_train = calc_mse_loss(df_train) - calc_mse_loss(df_train.append(pd.DataFrame(group), ignore_index=True))
        mse_loss_diff_val = calc_mse_loss(df_val) - calc_mse_loss(df_val.append(pd.DataFrame(group), ignore_index=True))
        mse_loss_diff_test = calc_mse_loss(df_test) - calc_mse_loss(df_test.append(pd.DataFrame(group), ignore_index=True))

        total_records = len(df_train) + len(df_val) + len(df_test)

        len_diff_train = (train_proportion - (len(df_train)/total_records))
        len_diff_val = (val_test_proportion - (len(df_val)/total_records))
        len_diff_test = (val_test_proportion - (len(df_test)/total_records))

        len_loss_diff_train = len_diff_train * abs(len_diff_train)
        len_loss_diff_val = len_diff_val * abs(len_diff_val)
        len_loss_diff_test = len_diff_test * abs(len_diff_test)

        loss_train = (hparam_mse_wgt * mse_loss_diff_train) + ((1-hparam_mse_wgt) * len_loss_diff_train)
        loss_val = (hparam_mse_wgt * mse_loss_diff_val) + ((1-hparam_mse_wgt) * len_loss_diff_val)
        loss_test = (hparam_mse_wgt * mse_loss_diff_test) + ((1-hparam_mse_wgt) * len_loss_diff_test)

        if (max(loss_train,loss_val,loss_test) == loss_train):
            df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
        elif (max(loss_train,loss_val,loss_test) == loss_val):
            df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
        else:
            df_test = df_test.append(pd.DataFrame(group), ignore_index=True)

        print ("Group " + str(i) + ". loss_train: " + str(loss_train) + " | " + "loss_val: " + str(loss_val) + " | " + "loss_test: " + str(loss_test) + " | ")
        i += 1

    return df_train, df_val, df_test