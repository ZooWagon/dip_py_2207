import jaydebeapi
inputFilePath="/soft/dip2207_back/input/"
outputFilePath="/soft/dip2207_back/output/"
url = 'jdbc:postgresql://192.168.0.237:26000/dip2207'
user = 'winston'
password = 'gauss@123'
dirver = 'org.postgresql.Driver'
jarFile = '/soft/dip2207_back/py/postgresql.jar'
# sqlStr = "select * from submission;"
# conn = jaydebeapi.connect(dirver, url, [user, password], jarFile)
# curs = conn.cursor()
# curs.execute(sqlStr)
# # result = curs.fetchall()
# # print(result)
# curs.close()
# conn.close()


def updateDBwithFinishSignal(sid):
    # update db
    conn = jaydebeapi.connect(dirver, url, [user, password], jarFile)
    curs = conn.cursor()
    sqlStr = "update submission set status='已完成' where sid='"+sid+"' ;"
    curs.execute(sqlStr)
    # result = curs.fetchall()
    # print(result)
    curs.close()
    conn.close()