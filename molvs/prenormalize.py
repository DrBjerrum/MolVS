"""
molvs.prenormalize
~~~~~~~~~~~~~~~

This module contains tools for prenormalizing and correcting common drawing errors using smarts patterns and dictionaries of A tag values.
It works on unsanitized RDMol objects and can thus be used to correct common drawing errors which lead to errorneous or unsanitizable RDMol's

:copyright: Copyright 2015 by Esben Jannik Bjerrum, Adapted from work Copyrigth 2014 by Matt Swain
:license: MIT, see LICENSE file for more details.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import logging
import numpy #needed by coordscaling
#import copy

from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import memoized_property


log = logging.getLogger(__name__)


class PreNormalization(object):
    """A PreNormalization transform defined by reaction SMARTS."""

    def __init__(self, name, transform, vtag=None):
        """
        :param string name: A name for this Normalization
        :param string transform: Reaction SMARTS to define the transformation
	:param string vtag: The V tag to apply to the atom, default none
        """
        log.debug('Initializing Normalization: %s', name)
        self.name = name
        self.transform_str = transform
	self.vtag = vtag

    @memoized_property
    def transform(self):
        log.debug('Loading PreNormalization transform: %s', self.name)
	transform = AllChem.ReactionFromSmarts(self.transform_str.encode('utf8'))
	if not self.vtag == None:
		transform.GetProducts()[0].GetAtomWithIdx(0).SetProp('molFileValue'.encode('utf8'),self.vtag.encode('utf8'))
        return transform

    def __repr__(self):
        return 'PreNormalization({!r}, {!r})'.format(self.name, self.transform_str)

    def __str__(self):
        return self.name


class AliasNormalization(object):
    """A AliasNormalization defined by alias and smiles"""

    def __init__(self, name, smiles, attach=1):
        """
        :param string name: A name for this Normalization
        :param string smiles: Reaction SMARTS to define the transformation.
	:param integer: Attachment point of smiles.
        """
        log.debug('Initializing Normalization: %s', name)
        self.name = name
        self.smiles_str = smiles
	self.attach = attach

    @memoized_property
    def smiles(self):
        log.debug('Loading AliasNormalization: %s', self.name)
        return AllChem.MolFromSmiles(self.smiles_str.encode('utf8'))

    def __repr__(self):
        return 'AliasNormalization({!r}, {!r},{!r})'.format(self.name, self.smiles_str, self.attach)

    def __str__(self):
        return self.name


#: The default list of Normalization transforms.
PRENORMALIZATIONS = (
	PreNormalization(u'Uncharged Tetravalent Nitrogen',u'[Nv4+0:1]>>[N+:1]'),
	PreNormalization(u'+4 Charged NH => Ammonium',u'[NH7+4]>>[N+:1]'),
	PreNormalization(u'Pentavalent carbon with methyl group',u'[Cv5:1][CH3]>>[Cv4:1]','pruned CH3')
)

ALIASNORMALIZATIONS = (
	AliasNormalization('OH', '[OH]',1),
	AliasNormalization('-O', '[O-]',1),
	AliasNormalization('ONa', '[O][Na]',1),
	AliasNormalization('NaO', '[O][Na]',1),
	AliasNormalization('HCl', '[ClH]',1),
	AliasNormalization('HBr', '[BrH]',1),
)


#: The default value for the maximum number of times to attempt to apply the series of normalizations.
MAX_RESTARTS = 200

class PreNormalizer(object):
    """A class for applying Normalization transforms.

    This class is typically used to apply a series of PreNormalization transforms to correct functional groups and
    recombine charges. Each transform is repeatedly applied until no further changes occur.
    """

    def __init__(self, prenormalizations=PRENORMALIZATIONS, aliasnormalizations=ALIASNORMALIZATIONS, max_restarts=MAX_RESTARTS):
        """Initialize a Normalizer with an optional custom list of :class:`~molvs.normalize.Normalization` transforms.

        :param normalizations: A list of  :class:`~molvs.normalize.Normalization` transforms to apply.
        :param int max_restarts: The maximum number of times to attempt to apply the series of normalizations (default
                                 200).
        """
        log.debug('Initializing PreNormalizer')
        self.prenormalizations = prenormalizations
        self.aliasnormalizations = aliasnormalizations
        self.aliases = {alias.name: alias.smiles_str for alias in self.aliasnormalizations}
        self.max_restarts = max_restarts

    def __call__(self, mol):
        """Calling a PreNormalizer instance like a function is the same as calling its prenormalize(mol) method."""
        return self.prenormalize(mol)

    def scale_coords(self, mol, target=1.5):
	""" Check if median bond length is within 2% of the target, otherwise scale coords of mol.
	RDkit has a default of 1.5, where other drawing programs produce different lengths (e.g. ChemDraw 1)

	:param mol: the rdkit mol to scale
	:param target: The target median bondlength. Default is 1.5"""
	#Compute Median Bond Length
	bondlengths = []
	conf = mol.GetConformer()
	for bond in mol.GetBonds():
		startatomidx = bond.GetBeginAtomIdx()
		endatomidx = bond.GetEndAtomIdx()
		lenght = AllChem.GetBondLength(conf, startatomidx, endatomidx)
		bondlengths.append(lenght)
	factor = target/numpy.median(bondlengths)
	#Scale coords if too divergent bondlength from RDkit 1.5
	if (factor < 0.98) or (factor > 1.02):
		log.info('Scaling original coords with factor %s'%(str(factor)))
		center = AllChem.ComputeCentroid(mol.GetConformer())
		tf = numpy.identity(4,numpy.float)
		tf[0][3] -= center[0]
		tf[1][3] -= center[1]
		tf[0][0] = tf[1][1] = tf[2][2] = factor
		AllChem.TransformMol(mol,tf)

    def prenormalize(self, mol):
        """Apply a series of Pre Normalization transforms to correct functional groups and recombine charges.

        A series of transforms are applied to the molecule. For each Normalization, the transform is applied repeatedly
        until no further changes occur. If any changes occurred, we go back and start from the first Normalization
        again, in case the changes mean an earlier transform is now applicable. The molecule is returned once the entire
        series of Normalizations cause no further changes or if max_restarts (default 200) is reached.

        :param mol: The molecule to normalize.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The normalized fragment.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        log.debug('Running PreNormalizer')
        # Normalize each fragment separately to get around quirky RunReactants behaviour
        fragments = []
	#Update Property Cache to allow transformations
	mol.UpdatePropertyCache(strict=False)
        for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
	    fragment.UpdatePropertyCache(strict=False) #Instead of Sanitazion
            fragments.append(self._normalize_fragment(fragment))
        # Join normalized fragments into a single molecule again
        outmol = fragments.pop()
        for fragment in fragments:
            outmol = Chem.CombineMols(outmol, fragment)
        #Chem.SanitizeMol(outmol)
        return outmol

    def _has_alias(self, mol):
	dic = {}
	for i,atom in enumerate(mol.GetAtoms()):
		if atom.HasProp('molFileAlias'.encode('ascii')):
			alias = atom.GetProp('molFileAlias'.encode('ascii'))
			log.debug("Atom Number %s, %s, has alias %s"%(i, atom.GetSymbol(),alias)) 
			dic[i] = alias
	return dic

	

    def substAlias_old(self, mol):
	dummy = Chem.MolFromSmiles('*') #TODO, CHeck if mol has dummy already???
	#Substitue 1
	#dic = self._has_alias(mol) # Builds dic of aliases observed with atom index. (Todo? What if IDX changes due to subst??)
	for i,atom in enumerate(mol.GetAtoms()): #Iterate through Atoms
		if atom.HasProp('molFileAlias'.encode('ascii')):
			alias = atom.GetProp('molFileAlias'.encode('ascii'))
			log.debug("Atom Number %s, %s, has alias %s"%(i, atom.GetSymbol(),alias))
			if self.aliases.has_key(alias):#If alias know, apply transformation
				atom.SetAtomicNum(0)
				repl = Chem.MolFromSmiles(self.aliases[alias],sanitize=False)
				template = AllChem.DeleteSubstructs(mol,dummy) # For optimizing coords of the new substitution
				#Chem.SanitizeMol(template)
				#Remove the alias from the atom so it doesn't get substituted multiple times
				atom.ClearProp('molFileAlias'.encode('ascii')) #TODO: Is the alias removed from the atom?
				log.debug('Removed:%s'%(atom.GetProp('molFileAlias'.encode('ascii'))))
				mol = AllChem.ReplaceSubstructs(mol,dummy,repl,True)[0] #Just chooses first product, What if multiple Aliases found?
				log.info("replaced alias %s with structure %s"%(alias, self.aliases[alias]))			
				#Optimize Coords
				AllChem.GenerateDepictionMatching2DStructure(mol,template)#TODO need to scale coords of template to match RDkits default
			elif idx >= 0:
				log.warn('Warning: Unknown atom alias found %s'%(alias))
	return mol

    def substAlias(self, mol):
	#Ensure that dummy do not already exist in mol, otherwise try another isotope.
	for dummytype in xrange(self.max_restarts):
		dummy = Chem.MolFromSmarts('[%s*]'%(str(dummytype)))
		#dummy.SetIsotope(dummytype+1) #TODO, CHeck if mol has dummy already
		if not mol.HasSubstructMatch(dummy):
			break
	else:
		log.error("Giving up finding unique dummy type number. Aborting alias prenormalization") #Hardly Probable
		return None
#	#Generate Template (Can it be done without doing two passes?
#	template = copy.deepcopy(mol)
#	#Remove all atoms with alias
#	dic = self._has_alias(template)
#	for idx in dic.keys():
#		atom = template.GetAtomWithIdx(idx)
#		atom.SetAtomicNum(0)
#	template = AllChem.DeleteSubstructs(template,dummy)
	#Go through atoms with Alias's
	scaledmol = False
	for n in xrange(self.max_restarts):
		for i,atom in enumerate(mol.GetAtoms()): #Iterate through Atoms, unfortunately has to be done again after each transformation 
			if atom.HasProp('molFileAlias'.encode('ascii')):
				alias = atom.GetProp('molFileAlias'.encode('ascii'))
				log.debug("Atom Number %s, %s, has alias %s"%(i, atom.GetSymbol(),alias))
				
				if self.aliases.has_key(alias):#If alias know, apply transformation
					if scaledmol == False:
						self.scale_coords(mol)
						scaledmol = True
						log.debug("Scaled coordinates to allow for RDkit optimization of depiction")
					atom.SetAtomicNum(0)
					atom.SetIsotope(dummytype)
					repl = Chem.MolFromSmiles(self.aliases[alias],sanitize=False)
					#Prepare a template for keeping known depiction coords
					template = AllChem.DeleteSubstructs(mol,dummy)
					#Remove the alias from the atom so it doesn't get substituted multiple times
					atom.ClearProp('molFileAlias'.encode('ascii')) # Removes the property
					log.debug('Removed:%s'%(atom.GetProp('molFileAlias'.encode('ascii'))))
					mol = AllChem.ReplaceSubstructs(mol,dummy,repl,True)[0] #TODO: Test if it works with multiple times alias in Mol.
					log.info("replaced alias %s with proper structure %s"%(alias, self.aliases[alias]))
					#Optimize Coords in depiction
					AllChem.GenerateDepictionMatching2DStructure(mol,template,acceptFailure=True)#TODO:This sometimes fail if multiple Alias's exist.
					break #Go back and restart main loop
				else:
					log.warn("Unknown Alias found: %s"%(alias))
		else:
			return mol
	# If we're still going after max_restarts (default 200), stop and warn, but still return the mol
        log.warn('Gave up alias substitution after %s restarts', self.max_restarts)
        return mol	

    def _normalize_fragment(self, mol):
        for n in xrange(self.max_restarts):
            # Iterate through Normalization transforms and apply each in order
            for normalization in self.prenormalizations:
                product = self._apply_transform(mol, normalization.transform)
                if product:
                    # If transform changed mol, go back to first rule and apply each again
                    log.info('Rule applied: %s', normalization.name)
                    mol = product
                    break
            else:
                # For loop finishes normally, all applicable transforms have been applied
                return mol
        # If we're still going after max_restarts (default 200), stop and warn, but still return the mol
        log.warn('Gave up normalization after %s restarts', self.max_restarts)
        return mol

    def _apply_transform(self, mol, rule):
        """Repeatedly apply pre normalization transform to molecule until no changes occur.

        It is possible for multiple products to be produced when a rule is applied. The rule is applied repeatedly to
        each of the products, until no further changes occur or after 20 attempts. If there are multiple unique products
        after the final application, the first product (sorted alphabetically by SMILES) is chosen.
        """
        mols = [mol]
        for n in xrange(20):
            products = {}
            for mol in mols:
		#Get all reactants
                for product in [x[0] for x in rule.RunReactants((mol,))]:                 
			#if Chem.SanitizeMol(product, catchErrors=True) == 0:
			product.UpdatePropertyCache(strict=False)
	                products[Chem.MolToSmiles(product, isomericSmiles=True)] = product
            if products:
                mols = [products[s] for s in sorted(products)]
            else:
                # If n == 0, the rule was not applicable and we return None
                return mols[0] if n > 0 else None

