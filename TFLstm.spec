# -*- mode: python -*-

block_cipher = None


a = Analysis(['TFLstm.py'],
             pathex=['C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\scipy\\extra-dll', 'D:\\audio\\proj\\MusicLevelAutoGen'],
             binaries=[],
             datas= [
                 ('C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\resampy\\data\\kaiser_best.npz', 'resampy\\data'),
                 ('C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\resampy\\data\\kaiser_fast.npz', 'resampy\\data')
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
