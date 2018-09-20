# -*- mode: python -*-

block_cipher = None

site_packages_dir = 'C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\'

a = Analysis(['TFLstm.py'],
             pathex=[site_packages_dir + 'scipy\\extra-dll', '.\\'],
             binaries=[],
             datas= [
                 (site_packages_dir + 'resampy\\data\\kaiser_best.npz', 'resampy\\data'),
                 (site_packages_dir + 'resampy\\data\\kaiser_fast.npz', 'resampy\\data'),
                 (site_packages_dir + 'tensorflow\\contrib\\rnn\\python\\ops\\_gru_ops.dll', 'tensorflow\\contrib\\rnn\\python\\ops'), 
                 (site_packages_dir + 'tensorflow\\contrib\\rnn\\python\\ops\\_lstm_ops.dll', 'tensorflow\\contrib\\rnn\\python\\ops')
             ],
             hiddenimports=[
                 'scipy._lib.messagestream',
                 'sklearn.neighbors.typedefs',
                 'sklearn.tree',
                 'sklearn.neighbors.kd_tree',
                 'sklearn.neighbors.quad_tree',
                 'sklearn.neighbors.ball_tree',
                 'sklearn.tree._utils',
                 'sklearn.tree._splitter',
                 'sklearn.tree._criterion',
                 'sklearn.tree._tree'],
             hookspath=['.\\pyinstaller_hook'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='TFLstm',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
