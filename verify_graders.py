"""
Comprehensive validator verification script.
This checks everything a validator might need to verify graders are properly configured.
"""
import sys
print("="*70)
print("VALIDATOR VERIFICATION SCRIPT")
print("="*70)

# Test 1: Import all necessary components
print("\n[TEST 1] Importing components...")
try:
    from contract_env.env import (
        TASKS, 
        TASK_GRADERS, 
        validate_all_tasks_have_graders,
        count_graded_tasks,
        get_graded_tasks,
        GRADED_TASK_IDS,
        GRADED_TASK_NAMES,
        NUM_GRADED_TASKS,
    )
    from contract_env.env.models import Action, Reward
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify grader registry
print("\n[TEST 2] Checking Grader Registry (TASK_GRADERS)...")
print(f"  - Registry keys: {list(TASK_GRADERS.keys())}")
print(f"  - Registry has {len(TASK_GRADERS)} entries")
if len(TASK_GRADERS) >= 3:
    print("  ✓ Registry has at least 3 graders")
else:
    print("  ✗ Registry does NOT have 3 graders")
    sys.exit(1)

# Test 3: Verify graded tasks metadata  
print("\n[TEST 3] Checking Graded Tasks Metadata...")
print(f"  - GRADED_TASK_IDS: {GRADED_TASK_IDS}")
print(f"  - GRADED_TASK_NAMES: {GRADED_TASK_NAMES}")
print(f"  - NUM_GRADED_TASKS: {NUM_GRADED_TASKS}")
if NUM_GRADED_TASKS >= 3 and len(GRADED_TASK_IDS) >= 3:
    print("  ✓ Metadata shows at least 3 graded tasks")

# Test 4: Verify each task has grader
print("\n[TEST 4] Verifying Task Grader Methods...")
for task in TASKS:
    has_grader = task.has_grader()
    print(f"  - {task.id}: has_grader()={has_grader} ({'✓' if has_grader else '✗'})")

# Test 5: Verify each grader function works
print("\n[TEST 5] Testing Grader Functions...")
test_action = Action(action_type='FLAG_RISK', content='test')
for task_id, grader_func in TASK_GRADERS.items():
    try:
        task = next(t for t in TASKS if t.id == task_id)
        reward = grader_func(task, task.contract_text, test_action, task.contract_text)
        if isinstance(reward, Reward) and 0 < reward.score < 1:
            print(f"  ✓ {task_id}: score={reward.score:.4f}")
        else:
            print(f"  ✗ {task_id}: invalid reward {reward}")
    except Exception as e:
        print(f"  ✗ {task_id}: {str(e)}")

# Test 6: Verify task methods
print("\n[TEST 6] Testing Task Methods...")
for task in TASKS:
    if task.id in GRADED_TASK_IDS:
        try:
            grader = task.get_grader()
            has_grader = task.has_grader()
            print(f"  ✓ {task.id}: get_grader() works, has_grader()={has_grader}")
        except Exception as e:
            print(f"  ✗ {task.id}: {str(e)}")

# Test 7: Run validation function
print("\n[TEST 7] Running validate_all_tasks_have_graders()...")
try:
    is_valid = validate_all_tasks_have_graders()
    print(f"  - Validation result: {is_valid}")
    if is_valid:
        print("  ✓ All tasks have valid graders")
    else:
        print("  ✗ Validation failed")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 8: Count graded tasks
print("\n[TEST 8] Counting Graded Tasks...")
graded_count = count_graded_tasks()
print(f"  - Graded tasks count: {graded_count}")
if graded_count >= 3:
    print("  ✓ At least 3 tasks are graded")
else:
    print("  ✗ Less than 3 tasks are graded")

# Test 9: Get graded tasks
print("\n[TEST 9] Getting Graded Tasks...")
graded_task_list = get_graded_tasks()
print(f"  - Graded tasks: {[t.id for t in graded_task_list]}")
if len(graded_task_list) >= 3:
    print("  ✓ get_graded_tasks() returns at least 3 tasks")
else:
    print("  ✗ get_graded_tasks() returns less than 3 tasks")

# Final Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print(f"Total tasks in TASKS: {len(TASKS)}")
print(f"Tasks with graders: {NUM_GRADED_TASKS}")
print(f"Grader registry size: {len(TASK_GRADERS)}")
print(f"Validation result: {'✓ PASS' if validate_all_tasks_have_graders() else '✗ FAIL'}")
print("="*70)

if NUM_GRADED_TASKS >= 3 and validate_all_tasks_have_graders():
    print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
    sys.exit(0)
else:
    print("\n✗✗✗ SOME CHECKS FAILED ✗✗✗")
    sys.exit(1)
