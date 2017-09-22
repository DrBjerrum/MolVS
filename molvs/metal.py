# -*- coding: utf-8 -*-
"""
molvs.metal
~~~~~~~~~~~

This module contains tools for disconnecting metal atoms that are defined as covalently bonded to non-metals.

:copyright: Copyright 2014 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import logging

from rdkit import Chem

from .utils import memoized_property


log = logging.getLogger(__name__)


class Disconnect(object):
    """A Disconnect defined by SMARTS."""

    def __init__(self, name, disconnect):
        """
        :param string name: A name for this Normalization
        :param string transform: Reaction SMARTS to define the transformation.
        """
        log.debug('Initializing Normalization: %s', name)
        self.name = name
        self.disconnect_str = disconnect

    @memoized_property
    def disconnect(self):
        log.debug('Loading disconnect smarts: %s', self.name)
        return Chem.MolFromSmarts(self.disconnect_str.encode('utf8'))

    def __repr__(self):
        return 'Disconnect({!r}, {!r})'.format(self.name, self.disconnect_str)

    def __str__(self):
        return self.name

DISCONNECTS = (
	Disconnect("FDA metal-NOF", '[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi]~[N,O,F]'),
	Disconnect("FDA Metal-Nonmetal", "[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Al,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,Hf,Ta,W,Re,Os,Ir,Pt,Au]~[B,C,Si,P,As,Sb,S,Se,Te,Cl,Br,I,At]")
	)



class MetalDisconnector(object):
    """Class for breaking covalent bonds between metals and organic atoms under certain conditions."""

    def __init__(self, disconnects=DISCONNECTS):
        log.debug('Initializing MetalDisconnector')
        # Initialize SMARTS to identify relevant substructures
        # TODO: Use atomic numbers instead of element symbols in SMARTS to allow for isotopes?
        self.disconnects = disconnects
        self._free_metal = Chem.MolFromSmarts('[Li+0,Na+0,K+0,Mg+0,CaX0+0]'.encode('utf8'))
        self._carboxylic = Chem.MolFromSmarts('[CX3](=O)[OX2H1]'.encode('utf8'))

    def __call__(self, mol):
        """Calling a MetalDisconnector instance like a function is the same as calling its disconnect(mol) method."""
        return self.disconnect(mol)

    def disconnect(self, mol):
        """Break covalent bonds between metals and organic atoms under certain conditions.

        The algorithm works as follows:

        - Disconnect N, O, F from any metal.
        - Disconnect other non-metals from transition metals + Al (but not Hg, Ga, Ge, In, Sn, As, Tl, Pb, Bi, Po).
        - For every bond broken, adjust the charges of the begin and end atoms accordingly.

        :param mol: The input molecule.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The molecule with metals disconnected.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        log.info('Running MetalDisconnector')
        # TODO: In future, maybe insert zero-order complex/ionic/dative bonds instead of disconnecting?
        # Remove bonds that match SMARTS
        for disconnectrule in self.disconnects: #[self._metal_nof, self._metal_non] + self._pauling_limit:
            smarts = disconnectrule.disconnect
            pairs = mol.GetSubstructMatches(smarts)
            edmol = Chem.EditableMol(mol)
            orders = []
            for i, j in pairs:
                # TODO: Could get the valence contributions of the bond instead of GetBondTypeAsDouble?
                orders.append(int(mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()))
                edmol.RemoveBond(i, j)
            # Adjust neighbouring charges accordingly
            mol = edmol.GetMol()
            for n, (i, j) in enumerate(pairs):
                chg = orders[n]
                atom1 = mol.GetAtomWithIdx(i)
                atom1.SetFormalCharge(atom1.GetFormalCharge() + chg)
                atom2 = mol.GetAtomWithIdx(j)
                atom2.SetFormalCharge(atom2.GetFormalCharge() - chg)
                log.info('Removed covalent bond between %s and %s', atom1.GetSymbol(), atom2.GetSymbol())
        # Ionize a free neutral metal with a protonated carboxylic acid
        # TODO: Extend this to other acids?
        # TODO: Move to charge module?
        fms = mol.GetSubstructMatches(self._free_metal)
        carbs = mol.GetSubstructMatches(self._carboxylic)
        if len(fms) == len(carbs) > 0:
            for fm in fms:
                atom = mol.GetAtomWithIdx(fm[0])
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            for carb in carbs:
                atom = mol.GetAtomWithIdx(carb[2])
                atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        log.debug(Chem.MolToSmiles(mol))
        Chem.SanitizeMol(mol)
        return mol
